package li.joye.yakuyomi.engine

import li.joye.yakuyomi.nightread.Gray
import li.joye.yakuyomi.nightread.Mask
import li.joye.yakuyomi.nightread.NightReadInput
import li.joye.yakuyomi.nightread.NightReadParams
import li.joye.yakuyomi.nightread.NightTier
import li.joye.yakuyomi.nightread.TextRegion
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import javax.imageio.ImageIO

/**
 * [NightReadRenderer.streamTiers]（多檔版的去重核心，不碰 Bitmap）：產品兩檔 `[L2, L3]`（「標準」與「更多」；L3 含「更多」
 * 新規則 A2；keep 與 L2 相同、但有貼紙要塗時 nightread 照樣合成，由這裡逐像素去重）對照完整三檔。
 *  - 第一檔（L2）一定交圖，而且與三檔版的 L2 逐像素相同；
 *  - L3 交 null ⇔ 它與 L2 逐像素相同；交圖時與三檔版的 L3 逐像素相同。
 * 三檔版沒交圖的檔取上一個交出的檔（[resolve]）——那樣去重在建構上正確，由函式庫的 SharedTierTest 對單檔 render 守。
 * 兩條分支（L3 交圖／交 null）都要有頁走到，見 [twoTiersMatchThreeTierRun]。
 * fixture＝nightread 函式庫的頁 fixture（engine 的巢狀 submodule `yakuyomi-nightread`，同 SharedTierTest 的讀法）。
 */
class NightReadTiersTest {

    private val dir = File("../yakuyomi-nightread/nightread/src/test/resources/page")

    private fun file(name: String): File = File(dir, name).also {
        check(it.isFile) { "缺 fixture：${it.path}（engine 的 yakuyomi-nightread submodule 沒 checkout？）" }
    }

    private fun readGray(name: String): Gray {
        val img = ImageIO.read(file(name))
        val g = Gray(img.width, img.height)
        val raster = img.raster
        for (y in 0 until img.height) for (x in 0 until img.width) g.data[y * img.width + x] = raster.getSample(x, y, 0)
        return g
    }

    private fun readMask(name: String): Mask {
        val g = readGray(name)
        return Mask(g.w, g.h, BooleanArray(g.data.size) { g.data[it] > 127 })
    }

    private fun readRegions(name: String): List<TextRegion> =
        file(name).readText().lineSequence().filter { it.isNotBlank() }.map {
            val p = it.trim().split(" ").map(String::toInt)
            TextRegion(p[0], p[1], p[2], p[3])
        }.toList()

    private fun input(page: String) = NightReadInput(
        gray = readGray("${page}_gray.png"),
        seg = readMask("${page}_seg.png"),
        regions = readRegions("${page}_regions.txt"),
        charMask = readMask("${page}_char.png"),
        chroma = readGray("${page}_chroma.png"),
    )

    /** 一次 streamTiers：每檔（依 [tiers] 順序）交出的灰階壓成 1 B/px；沒交圖＝null。也守回呼順序與 ARGB 不透明。 */
    private fun run(page: String, inp: NightReadInput, tiers: List<NightTier>): List<ByteArray?> {
        val px = IntArray(inp.gray.w * inp.gray.h)
        val order = ArrayList<Int>()
        val out = arrayOfNulls<ByteArray>(tiers.size)
        NightReadRenderer.streamTiers(inp, tiers, NightReadParams(), px, debug = null) { k, emitted, _ ->
            order += k
            if (emitted) {
                assertTrue("$page/${tiers[k].key}：交出的 ARGB 要不透明灰", px.all { p -> (p ushr 24) == 0xFF })
                out[k] = ByteArray(px.size) { (px[it] and 0xFF).toByte() }
            }
        }
        assertEquals("$page 回呼依序、每檔一次", tiers.indices.toList(), order)
        assertNotNull("$page 第一檔（${tiers[0].key}）一定交圖", out[0])
        return out.toList()
    }

    /** 沒交圖的檔＝上一個交出的檔（解析成每檔實際要顯示的像素）。 */
    private fun resolve(out: List<ByteArray?>): List<ByteArray> {
        val r = ArrayList<ByteArray>(out.size)
        for (o in out) r += o ?: r.last()
        return r
    }

    /** 一頁：兩檔版對照三檔版；回傳 L3 有沒有交圖。 */
    private fun checkPage(page: String): Boolean {
        val inp = input(page)
        val full = run(page, inp, NightTier.entries)
        val fullResolved = resolve(full)
        val l2 = fullResolved[NightTier.L2.ordinal]
        val l3 = fullResolved[NightTier.L3.ordinal]

        val two = run(page, inp, listOf(NightTier.L2, NightTier.L3))
        assertArrayEquals("$page：兩檔版的 L2 與三檔版的 L2 逐像素相同", l2, two[0])
        val same = l2.contentEquals(l3)
        if (two[1] == null) {
            assertTrue("$page：L3 交 null 只能是與 L2 逐像素相同", same)
        } else {
            assertTrue("$page：L3 與 L2 逐像素相同卻沒去重", !same)
            assertArrayEquals("$page：兩檔版的 L3 與三檔版的 L3 逐像素相同", l3, two[1])
        }
        println("  $page：三檔=${full.map { if (it == null) "-" else "●" }} 兩檔=${two.map { if (it == null) "-" else "●" }}")
        return two[1] != null
    }

    /**
     * 四頁 fixture（demo04 7.2 MPx 不跑，免測試 JVM OOM）。兩條分支都要走到：至少一頁 L3 ≠ L2（交圖）、至少一頁 L3 ＝ L2
     * （交 null）——哪頁走哪條隨函式庫演算法變（ch34_011 在 PEAK_MAX 800 後 L2＝L3），所以只要求各至少一頁。
     */
    @Test
    fun twoTiersMatchThreeTierRun() {
        val emitted = listOf("ch34_011", "demo02", "demo05", "demo06").map(::checkPage)
        assertTrue("至少一頁 L3 ≠ L2（交圖分支）", emitted.any { it })
        assertTrue("至少一頁 L3 ＝ L2（交 null 分支）", emitted.any { !it })
    }

    @Test
    fun singleTierEmitsOnce() {
        val inp = input("demo05")
        val one = run("demo05", inp, listOf(NightTier.L3))
        assertEquals(1, one.size)
        // 只跑 L3 時它就是第一檔：一定交圖，且與三檔版的 L3 相同
        val l3 = resolve(run("demo05", inp, NightTier.entries))[NightTier.L3.ordinal]
        assertArrayEquals("demo05：單檔 L3 與三檔版的 L3 逐像素相同", l3, one[0])
    }
}
