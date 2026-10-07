package li.joye.yakuyomi.engine

import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.nl.languageid.LanguageIdentification
import com.google.mlkit.nl.languageid.LanguageIdentificationOptions
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.TranslatorOptions
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/** Token usage kept for binary/source compatibility with the existing pipeline. */
data class Usage(val promptTokens: Int, val completionTokens: Int)

/**
 * Free on-device translator with automatic source-language detection and Arabic output.
 *
 * The class intentionally keeps the historical [LlmTranslator] name and public API so
 * the reader app and the rest of the engine do not need a migration layer.
 *
 * Flow per text region:
 *  1. Identify the source language on device with ML Kit Language ID.
 *  2. Resolve that language to an ML Kit translation model.
 *  3. Download the required source -> Arabic model once, on demand.
 *  4. Translate locally. Subsequent use of the downloaded model works offline.
 *
 * No manga text is sent to DeepSeek/OpenAI/Groq or another cloud provider.
 *
 * Important: language identification happens after OCR. It can choose the correct
 * translation model only when the OCR model has already produced usable source text.
 */
class LlmTranslator(
    @Suppress("UNUSED_PARAMETER") apiKey: String,
    @Suppress("UNUSED_PARAMETER") private val cfg: TranslatorConfig = TranslatorConfig(),
    private val postProcess: ((String) -> String)? = null,
) : Translator {

    data class TranslateResult(
        val translations: List<String>,
        val usage: Usage? = null,
        val error: String? = null,
        val raw: String? = null,
    )

    // Delay ML Kit client creation until the first actual translation request.
    // This keeps engine construction lightweight and, more importantly, ensures a
    // ML Kit initialization problem is reported as a per-page translation failure
    // instead of aborting construction after the large NCNN models are already loaded.
    private val languageIdentifierDelegate = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        LanguageIdentification.getClient(
            LanguageIdentificationOptions.Builder()
                .setConfidenceThreshold(LANGUAGE_CONFIDENCE)
                .build(),
        )
    }
    private val languageIdentifier by languageIdentifierDelegate

    private val clients = LinkedHashMap<String, com.google.mlkit.nl.translate.Translator>()
    private val readyModels = HashSet<String>()
    private val clientsMutex = Mutex()

    var lastError: String? = null
        private set

    var lastRaw: String? = null
        private set

    suspend fun translateDetailed(queries: List<String>): TranslateResult {
        if (queries.isEmpty()) return TranslateResult(emptyList())

        var failed = 0
        val errors = ArrayList<String>()
        val translations = ArrayList<String>(queries.size)

        for (sourceRaw in queries) {
            val source = sourceRaw.trim()
            if (source.isEmpty()) {
                translations += sourceRaw
                continue
            }

            val translated = try {
                translateOne(source)
            } catch (t: Throwable) {
                failed++
                val message = "${t.javaClass.simpleName}: ${t.message}"
                errors += message
                EngineTrace.log("translate.error $message")
                Log.w(TAG, "Local translation failed for one region: $message")
                sourceRaw
            }

            translations += translated.ifBlank {
                failed++
                errors += "Blank translation result"
                sourceRaw
            }
        }

        // A partially failed page must NOT be committed as successfully translated.
        // Pipeline checks this error and keeps the original page for a later retry.
        val error = if (failed == 0) {
            null
        } else {
            "Local translation failed in $failed of ${queries.count { it.isNotBlank() }} regions: " +
                errors.distinct().joinToString(" | ")
        }

        return TranslateResult(
            translations = translations,
            usage = null,
            error = error,
            raw = null,
        )
    }

    override suspend fun translate(queries: List<String>): List<String> {
        val result = translateDetailed(queries)
        lastError = result.error
        lastRaw = result.raw
        return result.translations
    }

    private suspend fun translateOne(source: String): String {
        // Short English manga dialogue ("NO!", "I'M FINE.") is frequently classified
        // as another Latin language. Preserve English -> Arabic deterministically for
        // Latin-only speech, while leaving Japanese/Chinese/Korean to language ID.
        val hasCjk = source.any {
            it.code in 0x3040..0x30FF || it.code in 0x3400..0x9FFF || it.code in 0xAC00..0xD7AF
        }
        val latinOnly = !hasCjk && source.any { it in 'A'..'Z' || it in 'a'..'z' } &&
            source.none { it.isLetter() && it !in 'A'..'Z' && it !in 'a'..'z' }
        val sourceLanguage = if (latinOnly) {
            TranslateLanguage.ENGLISH
        } else {
            val detectedTag = languageIdentifier.identifyLanguage(source).awaitString()
            resolveSourceLanguage(detectedTag, source)
        }

        // The page is already Arabic: keep it unchanged and avoid a pointless model download.
        if (sourceLanguage == TranslateLanguage.ARABIC) return source

        val client = getOrCreateClient(sourceLanguage)
        ensureModelReady(sourceLanguage, client)

        return client.translate(source)
            .awaitString()
            .trim()
            .let { postProcess?.invoke(it) ?: it }
            .trim()
    }

    /**
     * Language ID returns BCP-47 codes (for example en, ja, ko, zh).
     * ML Kit Translation supports a smaller set, so unsupported/undetermined text
     * falls back conservatively to English. This preserves the previous build's
     * behavior while adding automatic Japanese/Chinese/Korean/etc. when supported.
     */
    private fun resolveSourceLanguage(detectedTag: String, source: String): String {
        val normalized = detectedTag.substringBefore('-').lowercase()
        val mapped = if (normalized == UNDETERMINED) null else TranslateLanguage.fromLanguageTag(normalized)

        if (mapped != null) {
            EngineTrace.log("translate.lang detected=$normalized mapped=$mapped")
            return mapped
        }

        // Script hints improve very short manga bubbles where Language ID may answer "und".
        val hinted = when {
            source.any { it.code in 0x3040..0x30FF } -> TranslateLanguage.JAPANESE
            source.any { it.code in 0xAC00..0xD7AF } -> TranslateLanguage.KOREAN
            source.any { it.code in 0x4E00..0x9FFF } -> TranslateLanguage.CHINESE
            source.any { it.code in 0x0600..0x06FF } -> TranslateLanguage.ARABIC
            else -> TranslateLanguage.ENGLISH
        }
        EngineTrace.log("translate.lang detected=$detectedTag fallback=$hinted")
        return hinted
    }

    private suspend fun getOrCreateClient(sourceLanguage: String): com.google.mlkit.nl.translate.Translator =
        clientsMutex.withLock {
            clients[sourceLanguage] ?: Translation.getClient(
                TranslatorOptions.Builder()
                    .setSourceLanguage(sourceLanguage)
                    .setTargetLanguage(TranslateLanguage.ARABIC)
                    .build(),
            ).also { clients[sourceLanguage] = it }
        }

    private suspend fun ensureModelReady(
        sourceLanguage: String,
        client: com.google.mlkit.nl.translate.Translator,
    ) {
        clientsMutex.withLock {
            if (sourceLanguage in readyModels) return

            val conditions = DownloadConditions.Builder().build()
            client.downloadModelIfNeeded(conditions).awaitUnit()
            readyModels += sourceLanguage
            Log.i(TAG, "ML Kit $sourceLanguage -> Arabic model is ready")
        }
    }

    fun closeLocalTranslator() {
        if (languageIdentifierDelegate.isInitialized()) {
            runCatching { languageIdentifier.close() }
        }
        synchronized(clients) {
            clients.values.forEach { runCatching { it.close() } }
            clients.clear()
            readyModels.clear()
        }
    }

    private suspend fun Task<Void>.awaitUnit(): Unit =
        suspendCancellableCoroutine { continuation ->
            addOnSuccessListener {
                if (continuation.isActive) continuation.resume(Unit)
            }
            addOnFailureListener { error ->
                if (continuation.isActive) continuation.resumeWithException(error)
            }
            addOnCanceledListener {
                if (continuation.isActive) continuation.cancel()
            }
        }

    private suspend fun Task<String>.awaitString(): String =
        suspendCancellableCoroutine { continuation ->
            addOnSuccessListener { result ->
                if (continuation.isActive) continuation.resume(result)
            }
            addOnFailureListener { error ->
                if (continuation.isActive) continuation.resumeWithException(error)
            }
            addOnCanceledListener {
                if (continuation.isActive) continuation.cancel()
            }
        }

    companion object {
        private const val TAG = "LocalArabicTranslator"
        private const val UNDETERMINED = "und"
        private const val LANGUAGE_CONFIDENCE = 0.35f
    }
}
