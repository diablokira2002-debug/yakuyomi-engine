package li.joye.yakuyomi.engine

/**
 * Conservative OCR/translation filtering for Arabic V2.
 *
 * Two stages are intentionally separated:
 *  1. [sourceCandidates] runs before translation and removes only obvious OCR garbage.
 *  2. [apply] runs after translation and accepts only results that are safe to erase/re-render.
 *
 * A rejected region must remain untouched on the original manga page. The pipeline therefore
 * inpaints/renders only regions returned by [apply].
 */
internal object TextFilter {

    /**
     * Pre-translation filter. Be deliberately conservative: a false positive here would lose a
     * real speech bubble, so we only reject text that is clearly not meaningful language.
     */
    fun sourceCandidates(regions: List<TextRegion>): List<TextRegion> =
        regions.filter { isUsefulSource(it.sourceText) }

    /** Post-translation filter: return only regions safe to erase and redraw. */
    fun apply(regions: List<TextRegion>, filterText: String? = null): List<TextRegion> {
        val re = filterText?.let { runCatching { Regex(it) }.getOrNull() }
        return regions.filter { !shouldFilter(it, re) }
    }

    internal fun isUsefulSource(value: String): Boolean {
        val t = normalize(value)
        if (t.isEmpty()) return false
        if (t.any { it == '\uFFFD' || Character.isISOControl(it) }) return false

        // Numbers / punctuation / decorative glyphs do not need translation and are common OCR noise.
        if (t.none { it.isLetter() }) return false

        // Very symbol-heavy OCR usually comes from artwork/SFX edges rather than a readable bubble.
        val visible = t.count { !it.isWhitespace() }
        if (visible >= 4) {
            val lettersOrDigits = t.count { it.isLetterOrDigit() }
            if (lettersOrDigits.toFloat() / visible < MIN_TEXTUAL_RATIO) return false
        }

        return true
    }

    private fun shouldFilter(region: TextRegion, re: Regex?): Boolean {
        val source = normalize(region.sourceText)
        val translated = normalize(region.translatedText)

        if (translated.isEmpty()) return true
        if (translated.any { it == '\uFFFD' || Character.isISOControl(it) }) return true
        if (translated.all { it.isDigit() }) return true
        if (re != null && re.containsMatchIn(translated)) return true

        // Unchanged output means translation failed or the text was already Arabic.
        if (source.equals(translated, ignoreCase = true)) return true

        // Arabic V2 target safety: if the source contains non-Arabic letters, a successful result
        // should contain at least one Arabic letter. This prevents English/Japanese leakage from
        // being used to erase the original bubble.
        val sourceHasLetters = source.any { it.isLetter() }
        val sourceAlreadyArabic = containsArabicLetter(source)
        if (sourceHasLetters && !sourceAlreadyArabic && !containsArabicLetter(translated)) return true

        // Reject extremely symbol-heavy translated output as a last safety net.
        val visible = translated.count { !it.isWhitespace() }
        if (visible >= 4) {
            val lettersOrDigits = translated.count { it.isLetterOrDigit() }
            if (lettersOrDigits.toFloat() / visible < MIN_TEXTUAL_RATIO) return true
        }

        return false
    }

    private fun containsArabicLetter(text: String): Boolean = text.any {
        if (!it.isLetter()) return@any false
        val cp = it.code
        cp in 0x0600..0x06FF ||
            cp in 0x0750..0x077F ||
            cp in 0x08A0..0x08FF ||
            cp in 0xFB50..0xFDFF ||
            cp in 0xFE70..0xFEFF
    }

    private fun normalize(value: String): String =
        value
            .replace('\u00A0', ' ')
            .replace(Regex("""[ \t\r\n]+"""), " ")
            .trim()

    private const val MIN_TEXTUAL_RATIO = 0.35f
}
