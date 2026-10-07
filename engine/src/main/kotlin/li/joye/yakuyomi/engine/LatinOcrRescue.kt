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
import kotlin.math.sqrt

/**
 * Latin/English OCR path for Arabic V2.
 *
 * Important design rule:
 * - NCNN manga OCR stays as fallback/multilingual OCR.
 * - When ML Kit can read a Latin speech block, its WHOLE block text becomes authoritative.
 *
 * This avoids the previous failure mode where a broken NCNN fragment (e.g. "icanfeel",
 * "Y-4ou", "uike") survived and was translated literally. ML Kit blocks also preserve
 * word spacing and multi-line bubble sentences much better than line-by-line replacement.
 */
class LatinOcrResult(
    val lines: List<TextLine>,
    /** Tight per-line rectangles used only to extend the removal mask. */
    val latinMaskRects: List<Rect>,
) {
    fun augmentMask(base: Bitmap): Bitmap {
        if (latinMaskRects.isEmpty()) return base
        val out = base.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(out)
        val paint = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.FILL
        }
        latinMaskRects.forEach { r ->
            val padX = max(2, (r.height() * 0.10f).toInt())
            val padY = max(1, (r.height() * 0.06f).toInt())
            canvas.drawRect(
                (r.left - padX).coerceAtLeast(0).toFloat(),
                (r.top - padY).coerceAtLeast(0).toFloat(),
                (r.right + padX).coerceAtMost(out.width).toFloat(),
                (r.bottom + padY).coerceAtMost(out.height).toFloat(),
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
        // Webtoon pages often have very narrow, tiny text. A single full-page ML Kit
        // request misses bubbles on these images. Run overlapping, upscaled tiles and
        // map their boxes back to ORIGINAL page coordinates before any inpainting.
        val candidates = try {
            recognizeCandidates(page)
        } catch (t: Throwable) {
            EngineTrace.log("ocr.latin.error ${t.javaClass.simpleName}: ${t.message}")
            return LatinOcrResult(primary, emptyList())
        }

        if (candidates.isEmpty()) {
            EngineTrace.log("ocr.latin.done candidates=0 primary=${primary.size}")
            return LatinOcrResult(primary, emptyList())
        }

        val out = primary.toMutableList()
        val maskRects = mutableListOf<Rect>()
        var authoritativeBlocks = 0

        for (candidate in candidates) {
            val overlaps = out.mapIndexedNotNull { index, line ->
                val score = overlapScore(candidate.rect, line)
                if (score >= PRIMARY_MATCH_SCORE && line.direction == "h" && isMostlyLatin(line.text)) {
                    index to score
                } else {
                    null
                }
            }

            // A block is accepted when either:
            // 1) it overlaps a DBNet/NCNN Latin detection, OR
            // 2) DBNet missed it but it sits on a locally uniform speech/narration background.
            val accepted = overlaps.isNotEmpty() || looksLikeBubbleBackground(page, candidate.rect)
            if (!accepted) {
                EngineTrace.log("ocr.latin.reject_bg '${candidate.text.take(44)}'")
                continue
            }

            // ML Kit's whole block is authoritative for Latin dialogue. Remove every overlapping
            // primary Latin line first so broken fragments can never be grouped alongside it.
            if (overlaps.isNotEmpty()) {
                overlaps.map { it.first }.sortedDescending().forEach { out.removeAt(it) }
            }

            val r = candidate.rect
            out += TextLine(
                listOf(
                    Pt(r.left.toFloat(), r.top.toFloat()),
                    Pt(r.right.toFloat(), r.top.toFloat()),
                    Pt(r.right.toFloat(), r.bottom.toFloat()),
                    Pt(r.left.toFloat(), r.bottom.toFloat()),
                ),
                0.99f,
            ).apply {
                direction = "h"
                text = candidate.text
            }

            // Use each original ML Kit line box for text removal instead of filling the entire
            // speech block rectangle. This erases the English glyph area without wiping the bubble.
            if (candidate.lineRects.isNotEmpty()) maskRects += candidate.lineRects else maskRects += Rect(r)

            authoritativeBlocks++
            EngineTrace.log("ocr.latin.authoritative '${candidate.text.take(64)}'")
        }

        EngineTrace.log(
            "ocr.latin.done candidates=${candidates.size} authoritative=$authoritativeBlocks " +
                "primary=${primary.size}->${out.size}",
        )
        return LatinOcrResult(out, maskRects)
    }

    private suspend fun recognizeCandidates(page: Bitmap): List<Candidate> {
        val raw = mutableListOf<Candidate>()
        try {
            val full = recognizer.process(InputImage.fromBitmap(page, 0)).awaitText()
            raw += parseCandidates(full, 0, 1f, page.width, page.height)
        } catch (t: Throwable) {
            EngineTrace.log("ocr.latin.full.error ${t.javaClass.simpleName}: ${t.message}")
        }

        // Small-width webtoon screenshots (about 300px) particularly benefit from
        // 2x enlargement. Large regular manga pages use their original-resolution scan.
        if (page.width < 900 && page.height > 700) {
            val scale = if (page.width < 450) 2f else 1.5f
            val tileHeight = (1000f / scale).toInt().coerceAtLeast(360)
            val step = (tileHeight * 0.76f).toInt().coerceAtLeast(1)
            var top = 0
            while (top < page.height) {
                val bottom = (top + tileHeight).coerceAtMost(page.height)
                val tile = Bitmap.createBitmap(page, 0, top, page.width, bottom - top)
                val scaled = Bitmap.createScaledBitmap(
                    tile,
                    (tile.width * scale).toInt(),
                    (tile.height * scale).toInt(),
                    true,
                )
                try {
                    val result = recognizer.process(InputImage.fromBitmap(scaled, 0)).awaitText()
                    raw += parseCandidates(result, top, scale, page.width, page.height)
                } catch (t: Throwable) {
                    EngineTrace.log("ocr.latin.tile.error top=$top ${t.javaClass.simpleName}: ${t.message}")
                } finally {
                    scaled.recycle()
                    tile.recycle()
                }
                if (bottom == page.height) break
                top += step
            }
        }

        // The same bubble can appear in the full-page result and overlapping tiles.
        // Preserve the longest recognized text and only one removal mask per bubble.
        val deduplicated = mutableListOf<Candidate>()
        for (candidate in raw.sortedByDescending { it.text.length }) {
            val duplicate = deduplicated.any { prior ->
                val overlap = rectOverlap(candidate.rect, prior.rect)
                overlap >= 0.70f &&
                    (
                        candidate.text.equals(prior.text, ignoreCase = true) ||
                            candidate.text.contains(prior.text, ignoreCase = true) ||
                            prior.text.contains(candidate.text, ignoreCase = true) ||
                            overlap >= 0.92f
                        )
            }
            if (!duplicate) deduplicated += candidate
        }
        EngineTrace.log("ocr.latin.scan raw=${raw.size} unique=${deduplicated.size}")
        return deduplicated.sortedWith(compareBy({ it.rect.top }, { it.rect.left }))
    }

    private fun parseCandidates(
        output: Text,
        tileTop: Int,
        scale: Float,
        width: Int,
        height: Int,
    ): List<Candidate> {
        fun original(r: Rect): Rect = Rect(
            (r.left / scale).toInt().coerceIn(0, width),
            (tileTop + r.top / scale).toInt().coerceIn(0, height),
            (r.right / scale).toInt().coerceIn(0, width),
            (tileTop + r.bottom / scale).toInt().coerceIn(0, height),
        )

        return output.textBlocks.mapNotNull { block ->
            val rect = block.boundingBox?.let(::original) ?: return@mapNotNull null
            val text = normalize(block.text)
            if (!isUsefulLatin(text) || isLikelyWatermark(text)) return@mapNotNull null
            if (rect.width() < 6 || rect.height() < 5) return@mapNotNull null
            val lineRects = block.lines.mapNotNull { it.boundingBox?.let(::original) }
                .filter { it.width() >= 3 && it.height() >= 3 }
            Candidate(text, rect, lineRects)
        }
    }

    private fun rectOverlap(a: Rect, b: Rect): Float {
        val w = (minOf(a.right, b.right) - maxOf(a.left, b.left)).coerceAtLeast(0)
        val h = (minOf(a.bottom, b.bottom) - maxOf(a.top, b.top)).coerceAtLeast(0)
        val area = w.toLong() * h
        val smaller = minOf(a.width().toLong() * a.height(), b.width().toLong() * b.height())
        return if (smaller <= 0) 0f else area.toFloat() / smaller
    }

    override fun close() {
        if (recognizerDelegate.isInitialized()) {
            runCatching { recognizer.close() }
        }
    }

    private data class Candidate(
        val text: String,
        val rect: Rect,
        val lineRects: List<Rect>,
    )

    private fun isUsefulLatin(value: String): Boolean {
        val t = normalize(value)
        val letters = t.count { it.isLetter() }
        if (letters < 2) return false
        val latin = t.count { isLatinLetter(it) }
        if (latin.toFloat() / letters < MIN_LATIN_RATIO) return false

        // Reject OCR garbage mixing digits into words: Y-4ou, a12b, etc.
        if (Regex("""(?i)\b(?=[a-z0-9-]*[a-z])(?=[a-z0-9-]*\d)[a-z0-9-]{3,}\b""").containsMatchIn(t)) return false

        return true
    }

    private fun isMostlyLatin(value: String): Boolean {
        val t = normalize(value)
        val letters = t.count { it.isLetter() }
        if (letters == 0) return false
        return t.count { isLatinLetter(it) }.toFloat() / letters >= MIN_LATIN_RATIO
    }

    private fun isLatinLetter(ch: Char): Boolean =
        ch in 'A'..'Z' || ch in 'a'..'z' || ch.code in 0x00C0..0x024F

    private fun isLikelyWatermark(value: String): Boolean {
        val s = value.lowercase().replace(Regex("""\s+"""), "")
        if (Regex("""(?:https?://|www\.|[a-z0-9-]+\.(?:com|org|net|io|ink|co)\b)""").containsMatchIn(s)) return true
        if (listOf("manga", "comic", "scanlat", "scanlation", "webtoon", "manhwa").any { it in s } && s.length <= 48) return true
        if (s.endsWith("ink") && ("manga" in s || "comic" in s || "like" in s) && s.length <= 40) return true
        return false
    }

    /**
     * Accept DBNet-missed Latin blocks only on locally uniform backgrounds.
     * Works for white/cream speech balloons and uniform dark/purple balloons while rejecting most
     * site watermarks over detailed artwork.
     */
    private fun looksLikeBubbleBackground(page: Bitmap, rect: Rect): Boolean {
        val padX = (rect.width() * 0.30f).toInt().coerceAtLeast(8)
        val padY = (rect.height() * 0.45f).toInt().coerceAtLeast(8)
        val l = (rect.left - padX).coerceIn(0, page.width - 1)
        val t = (rect.top - padY).coerceIn(0, page.height - 1)
        val r = (rect.right + padX).coerceIn(l + 1, page.width)
        val b = (rect.bottom + padY).coerceIn(t + 1, page.height)

        val stepX = ((r - l) / 28).coerceAtLeast(1)
        val stepY = ((b - t) / 20).coerceAtLeast(1)
        var n = 0
        var sum = 0.0
        var sum2 = 0.0

        var y = t
        while (y < b) {
            var x = l
            while (x < r) {
                // Exclude the glyph rectangle itself.
                if (x !in rect.left..rect.right || y !in rect.top..rect.bottom) {
                    val p = page.getPixel(x, y)
                    val lum = 0.299 * ((p shr 16) and 0xFF) +
                        0.587 * ((p shr 8) and 0xFF) +
                        0.114 * (p and 0xFF)
                    sum += lum
                    sum2 += lum * lum
                    n++
                }
                x += stepX
            }
            y += stepY
        }
        if (n < 24) return false
        val mean = sum / n
        val variance = (sum2 / n - mean * mean).coerceAtLeast(0.0)
        return sqrt(variance) <= MAX_BUBBLE_LUMA_STD
    }

    /**
     * Intersection over the smaller box. More forgiving than IoU because DBNet often returns a
     * tight word/line box while ML Kit returns the whole text block.
     */
    private fun overlapScore(rect: Rect, line: TextLine): Float {
        val l = line.quad.minOf { it.x }.toInt()
        val t = line.quad.minOf { it.y }.toInt()
        val r = line.quad.maxOf { it.x }.toInt()
        val b = line.quad.maxOf { it.y }.toInt()
        val ix = (minOf(rect.right, r) - maxOf(rect.left, l)).coerceAtLeast(0)
        val iy = (minOf(rect.bottom, b) - maxOf(rect.top, t)).coerceAtLeast(0)
        val inter = ix * iy
        if (inter <= 0) return 0f
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
        private const val PRIMARY_MATCH_SCORE = 0.08f
        private const val MAX_BUBBLE_LUMA_STD = 62.0
    }
}
