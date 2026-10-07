package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Test

class GroupingTextJoinTest {

    private fun line(text: String): TextLine {
        val q = listOf(Pt(0f, 0f), Pt(100f, 0f), Pt(100f, 20f), Pt(0f, 20f))
        return TextLine(q, 1f).apply { this.text = text }
    }

    @Test
    fun joinsEnglishLinesWithSpaces() {
        val region = TextRegion(
            lines = listOf(line("THE COLORS"), line("ARE ALL"), line("WRONG")),
            direction = "h",
        )
        assertEquals("THE COLORS ARE ALL WRONG", region.sourceText)
    }

    @Test
    fun joinsJapaneseLinesWithoutSpaces() {
        val region = TextRegion(
            lines = listOf(line("その通り"), line("じゃ")),
            direction = "v",
        )
        assertEquals("その通りじゃ", region.sourceText)
    }

    @Test
    fun stripsSpaceBeforePunctuation() {
        val region = TextRegion(
            lines = listOf(line("ARE YOU"), line("OK ?")),
            direction = "h",
        )
        assertEquals("ARE YOU OK?", region.sourceText)
    }
}
