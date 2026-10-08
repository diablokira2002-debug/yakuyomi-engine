package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Test

/** Arabic V2 safety filters: do not erase a bubble unless OCR + translation are trustworthy enough. */
class TextFilterTest {

    private fun region(src: String, translated: String): TextRegion {
        val quad = listOf(Pt(0f, 0f), Pt(10f, 0f), Pt(10f, 10f), Pt(0f, 10f))
        val line = TextLine(quad, 1f).apply { text = src }
        return TextRegion(listOf(line), "h").apply { translatedText = translated }
    }

    @Test fun keepsNormalArabicTranslation() {
        assertEquals(1, TextFilter.apply(listOf(region("hello", "مرحبا"))).size)
    }

    @Test fun dropsBlankDigitAndUntranslated() {
        val kept = TextFilter.apply(
            listOf(
                region("hello", ""),
                region("123", "123"),
                region("hello", "hello"),
                region("cat", "قطة"),
            ),
        )
        assertEquals(1, kept.size)
        assertEquals("قطة", kept[0].translatedText)
    }

    @Test fun dropsNonArabicLeakageForNonArabicSource() {
        val kept = TextFilter.apply(
            listOf(
                region("hello", "bonjour"),
                region("hello", "مرحبا"),
            ),
        )
        assertEquals(1, kept.size)
        assertEquals("مرحبا", kept[0].translatedText)
    }

    @Test fun keepsOriginalWhenMachineTranslationLeaksOcrGarbage() {
        val kept = TextFilter.apply(
            listOf(
                region("very skilled", "ماهر جدا diteedeawt"),
                region("this time", "هذه المرة"),
            ),
        )
        assertEquals(1, kept.size)
        assertEquals("هذه المرة", kept[0].translatedText)
    }

    @Test fun dropsRegexMatch() {
        val kept = TextFilter.apply(
            listOf(region("x", "إعلان مدفوع"), region("y", "نص طبيعي")),
            filterText = "إعلان",
        )
        assertEquals(1, kept.size)
        assertEquals("نص طبيعي", kept[0].translatedText)
    }

    @Test fun sourceFilterDropsObviousGarbageButKeepsRealText() {
        val sources = listOf(
            region("!!!", ""),
            region("1234", ""),
            region("A@#$%^", ""),
            region("HELLO!", ""),
            region("その通り", ""),
        )
        val kept = TextFilter.sourceCandidates(sources)
        assertEquals(listOf("HELLO!", "その通り"), kept.map { it.sourceText })
    }
}
