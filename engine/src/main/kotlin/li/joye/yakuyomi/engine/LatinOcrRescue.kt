package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.max

/**
 * Latin/English OCR rescue for Arabic V2.
 *
 * The NCNN manga OCR remains the primary recognizer because it supports the manga model's
 * multilingual alphabet. ML Kit Latin OCR runs once per page and is used conservatively to:
 *  - replace obviously broken Latin OCR on an already-detected text line, and
 *  - recover a horizontal Latin line that DBNet/NCNN OCR missed entirely.
 *
 * Added lines carry a tight rectangular rescue mask so the original Latin text can be removed
 * before Arabic is rendered. URL/watermark-like text is intentionally ignored.
 */
class LatinOcrResult(
    val lines: List<TextLine>,
    val addedLatinRects: List<Rect>,
) {
    fun augmentMask(base: Bitmap): Bitmap {
        if (addedLatinRects.isEmpty()) return base
        val out = base.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(out)
        val paint = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.FILL
        }
        addedLatinRects.forEach { r ->
            canvas.drawRect(
                (r.left - 1).coerceAtLeast(0).toFloat(),
                (r.top - 1).coerceAtLeast(0).toFloat(),
                (r.right + 1).coerceAtMost(out.width).toFloat(),
                (r.bottom + 1).coerceAtMost(out.height).toFloat(),
                paint,
            )
        }
        return out
    }
}

internal class LatinOcrRescue : AutoCloseable {

    private val recognizerDelegate = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    }
    private val recognizer by recognizerDelegate

    suspend fun rescue(page: Bitmap, primary: List<TextLine>): LatinOcrResult {
        val mlText = try {
            recognizer.process(InputImage.fromBitmap(page, 0)).awaitText()
        } catch (t: Throwable) {
            EngineTrace.log("ocr.latin.error ${t.javaClass.simpleName}: ${t.message}")
            return LatinOcrResult(primary, emptyList())
        }

        val out = primary.toMutableList()
        val added = mutableListOf<Rect>()

        val candidates = mlText.textBlocks
            .flatMap { it.lines }
            .mapNotNull { line ->
                val rect = line.boundingBox ?: return@mapNotNull null
                val text = normalize(line.text)
                if (!isUsefulLatin(text) || isLikelyWatermark(text)) return@mapNotNull null
                if (rect.width() < 6 || rect.height() < 6) return@mapNotNull null
                Candidate(text, rect)
            }

        for (candidate in candidates) {
            val best = out
                .mapIndexed { index, line -> index to overlapScore(candidate.rect, line) }
                .maxByOrNull { it.second }

            if (best != null && best.second >= MATCH_SCORE) {
                val line = out[best.first]
                val old = normalize(line.text)
                if (line.direction == "h" && shouldReplace(old, candidate.text)) {
                    EngineTrace.log("ocr.latin.replace '${old.take(28)}' -> '${candidate.text.take(28)}'")
                    line.text = candidate.text
                }
                continue
            }

            // New DBNet-missed rescue lines: only accept clearly horizontal Latin dialogue.
            val horizontal = candidate.rect.width() >= candidate.rect.height() * MIN_HORIZONTAL_RATIO
            if (!horizontal) continue
            if (out.any { overlapScore(candidate.rect, it) >= NEW_LINE_DUPLICATE_SCORE }) continue

            val r = candidate.rect
            val rescued = TextLine(
                listOf(
                    Pt(r.left.toFloat(), r.top.toFloat()),
                    Pt(r.right.toFloat(), r.top.toFloat()),
                    Pt(r.right.toFloat(), r.bottom.toFloat()),
                    Pt(r.left.toFloat(), r.bottom.toFloat()),
                ),
                0.95f,
            ).apply {
                direction = "h"
                text = candidate.text
            }
            out += rescued
            added += Rect(r)
            EngineTrace.log("ocr.latin.add '${candidate.text.take(32)}'")
        }

        if (candidates.isNotEmpty()) {
            EngineTrace.log("ocr.latin.done candidates=${candidates.size} added=${added.size}")
        }
        return LatinOcrResult(out, added)
    }

    override fun close() {
        if (recognizerDelegate.isInitialized()) {
            runCatching { recognizer.close() }
        }
    }

    private data class Candidate(val text: String, val rect: Rect)

    private fun shouldReplace(old: String, rescue: String): Boolean {
        if (old.isBlank()) return true
        if (!isMostlyLatin(old)) return false
        if (!isUsefulLatin(rescue)) return false

        if (looksCorruptLatin(old)) return true

        val oldScore = latinQuality(old)
        val rescueScore = latinQuality(rescue)
        if (rescueScore >= oldScore + 3) return true

        // Typical manga OCR failure: words glued together or only a fragment survives.
        val oldLetters = old.count { it.isLetter() }
        val rescueLetters = rescue.count { it.isLetter() }
        if (rescueLetters >= max(4, (oldLetters * 1.30f).toInt()) && rescueScore >= oldScore) return true
        if (!old.contains(' ') && rescue.contains(' ') && rescueLetters >= oldLetters && rescueScore > oldScore) return true

        return false
    }

    private fun latinQuality(value: String): Int {
        val t = normalize(value)
        val letters = t.count { isLatinLetter(it) }
        val spaces = t.count { it == ' ' }.coerceAtMost(4)
        val digits = t.count { it.isDigit() }
        val weird = t.count { !it.isLetterOrDigit() && !it.isWhitespace() && it !in ".,!?;:'’\"-()…" }
        var score = letters * 2 + spaces * 2 - weird * 3
        if (Regex("""(?i)\b(?=[a-z0-9-]*[a-z])(?=[a-z0-9-]*\d)[a-z0-9-]{3,}\b""").containsMatchIn(t)) {
            score -= 8
        }
        if (digits > letters / 2 && letters > 0) score -= 4
        return score
    }

    private fun looksCorruptLatin(value: String): Boolean {
        val t = normalize(value)
        if (Regex("""(?i)\b(?=[a-z0-9-]*[a-z])(?=[a-z0-9-]*\d)[a-z0-9-]{3,}\b""").containsMatchIn(t)) return true
        if (t.length >= 7 && !t.contains(' ') && t.count { isLatinLetter(it) } >= 7) return true
        return false
    }

    private fun isUsefulLatin(value: String): Boolean {
        val t = normalize(value)
        val letters = t.count { it.isLetter() }
        if (letters < 2) return false
        val latin = t.count { isLatinLetter(it) }
        return latin.toFloat() / letters >= MIN_LATIN_RATIO
    }

    private fun isMostlyLatin(value: String): Boolean {
        val letters = value.count { it.isLetter() }
        if (letters == 0) return false
        return value.count { isLatinLetter(it) }.toFloat() / letters >= MIN_LATIN_RATIO
    }

    private fun isLatinLetter(ch: Char): Boolean =
        ch in 'A'..'Z' || ch in 'a'..'z' || ch.code in 0x00C0..0x024F

    private fun isLikelyWatermark(value: String): Boolean {
        val s = value.lowercase()
        if (Regex("""(?:https?://|www\.|[a-z0-9-]+\.(?:com|org|net|io|ink|co)\b)""").containsMatchIn(s)) return true
        if ((s.contains("manga") || s.contains("comic") || s.contains("scanlat")) && s.length <= 32) return true
        return false
    }

    private fun overlapScore(rect: Rect, line: TextLine): Float {
        val l = line.quad.minOf { it.x }.toInt()
        val t = line.quad.minOf { it.y }.toInt()
        val r = line.quad.maxOf { it.x }.toInt()
        val b = line.quad.maxOf { it.y }.toInt()
        val ix = (minOf(rect.right, r) - maxOf(rect.left, l)).coerceAtLeast(0)
        val iy = (minOf(rect.bottom, b) - maxOf(rect.top, t)).coerceAtLeast(0)
        val inter = ix * iy
        if (inter <= 0) {
            val cx = (rect.left + rect.right) / 2f
            val cy = (rect.top + rect.bottom) / 2f
            val pad = max(rect.height(), b - t) * 0.6f
            return if (cx in (l - pad)..(r + pad) && cy in (t - pad)..(b + pad)) 0.22f else 0f
        }
        val a = (rect.width() * rect.height()).coerceAtLeast(1)
        val c = ((r - l).coerceAtLeast(1) * (b - t).coerceAtLeast(1))
        return inter.toFloat() / minOf(a, c).coerceAtLeast(1)
    }

    private fun normalize(value: String): String =
        value
            .replace('\u00A0', ' ')
            .replace(Regex("""[\t\r\n ]+"""), " ")
            .trim()

    private suspend fun Task<Text>.awaitText(): Text =
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
        private const val MIN_LATIN_RATIO = 0.72f
        private const val MIN_HORIZONTAL_RATIO = 1.10f
        private const val MATCH_SCORE = 0.28f
        private const val NEW_LINE_DUPLICATE_SCORE = 0.12f
    }
}
