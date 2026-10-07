package li.joye.yakuyomi.engine

import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.TranslatorOptions
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.suspendCancellableCoroutine

/** Token usage kept for binary/source compatibility with the existing pipeline. */
data class Usage(val promptTokens: Int, val completionTokens: Int)

/**
 * Free local English -> Arabic translator.
 *
 * This class intentionally keeps the old [LlmTranslator] name and public API so the
 * reader app does not need to change immediately. It no longer sends manga text to
 * DeepSeek/Groq/OpenAI. Translation is performed by ML Kit on the device.
 *
 * The English and Arabic ML Kit language models are downloaded automatically the
 * first time translation is used. After that, translation works offline.
 *
 * Detection, OCR, text removal and re-rendering remain handled by Yakuyomi's local
 * pipeline exactly as before.
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

    private val modelMutex = Mutex()

    @Volatile
    private var modelReady = false

    private val client = Translation.getClient(
        TranslatorOptions.Builder()
            .setSourceLanguage(TranslateLanguage.ENGLISH)
            .setTargetLanguage(TranslateLanguage.ARABIC)
            .build(),
    )

    var lastError: String? = null
        private set

    var lastRaw: String? = null
        private set

    /**
     * Translate all text regions from one manga page locally.
     *
     * Failed individual regions fall back to their OCR source text so Yakuyomi never
     * destroys a page because one bubble could not be translated. If every region
     * fails, [error] is populated so the existing pipeline marks the page retryable.
     */
    suspend fun translateDetailed(queries: List<String>): TranslateResult {
        if (queries.isEmpty()) return TranslateResult(emptyList())

        return try {
            ensureModelReady()

            var failed = 0
            val translations = ArrayList<String>(queries.size)

            for (source in queries) {
                if (source.isBlank()) {
                    translations += source
                    continue
                }

                val translated = try {
                    client.translate(source).awaitString()
                        .trim()
                        .let { postProcess?.invoke(it) ?: it }
                        .trim()
                } catch (t: Throwable) {
                    Log.w(TAG, "Local translation failed for one region: ${t.message}")
                    failed++
                    source
                }

                translations += translated.ifBlank {
                    failed++
                    source
                }
            }

            val error = if (failed >= queries.size) {
                "ML Kit local translation failed for all ${queries.size} regions"
            } else {
                null
            }

            TranslateResult(
                translations = translations,
                usage = null,
                error = error,
                raw = null,
            )
        } catch (t: Throwable) {
            Log.e(TAG, "Local Arabic translation failed: ${t.message}", t)
            TranslateResult(
                translations = queries,
                usage = null,
                error = "${t.javaClass.simpleName}: ${t.message}",
                raw = null,
            )
        }
    }

    override suspend fun translate(queries: List<String>): List<String> {
        val result = translateDetailed(queries)
        lastError = result.error
        lastRaw = result.raw
        return result.translations
    }

    /**
     * Downloads the English/Arabic language models once when first needed.
     * No Wi-Fi-only restriction is used so it also works for users on mobile data.
     */
    private suspend fun ensureModelReady() {
        if (modelReady) return

        modelMutex.withLock {
            if (modelReady) return

            val conditions = DownloadConditions.Builder().build()
            client.downloadModelIfNeeded(conditions).awaitUnit()
            modelReady = true
            Log.i(TAG, "ML Kit English -> Arabic models are ready")
        }
    }

    /** Can be called later when Pipeline lifecycle is extended to close translators. */
    fun closeLocalTranslator() {
        runCatching { client.close() }
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
    }
}
