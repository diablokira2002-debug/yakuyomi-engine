package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * 低優先權（夜讀）上鎖路徑的掛鉤順序（[NcnnBackend.forward]，前向換成假的 block，不碰原生）。
 *
 * 守的是優先權倒置的窗口：[NcnnLowPriorityHook.aroundLockedForward] 要在取全域鎖**之前**進、放鎖**之後**出——
 * 夜讀在那裡把 nice 拉回 0，拿到鎖才進掛鉤的話，取鎖到拉回之間、還原到放鎖之間都是 nice 9 持鎖。
 */
class NcnnForwardHookTest {

    /** 記下每個回呼當下有沒有持鎖；[abortOnCall] 為第幾次 shouldAbort 回 true（1 起算，0＝永不）。 */
    private class RecordingHook(private val abortOnCall: Int = 0) : NcnnLowPriorityHook {
        val events = ArrayList<String>()
        private var abortCalls = 0

        override fun shouldAbort(): Boolean {
            abortCalls++
            events += "abort? locked=${NcnnBackend.holdsNcnnLock()}"
            return abortCalls == abortOnCall
        }

        override fun <T> aroundLockedForward(block: () -> T): T {
            events += "enter locked=${NcnnBackend.holdsNcnnLock()}"
            try {
                return block()
            } finally {
                events += "exit locked=${NcnnBackend.holdsNcnnLock()}"
            }
        }
    }

    @Test
    fun hookBracketsTheWholeLockHold() {
        val hook = RecordingHook()
        val r = NcnnBackend.forward(serialize = true, lowPriority = true, hook = hook) {
            hook.events += "forward locked=${NcnnBackend.holdsNcnnLock()}"
            42
        }
        assertEquals(42, r)
        assertEquals(
            listOf(
                "abort? locked=false", // 前向開始前
                "enter locked=false", // 取鎖之前就進掛鉤
                "abort? locked=true", // 拿到鎖後正式開跑前再問一次
                "forward locked=true",
                "exit locked=false", // 放鎖之後才出掛鉤
            ),
            hook.events,
        )
        assertFalse(NcnnBackend.holdsNcnnLock())
    }

    @Test
    fun abortAfterTakingTheLockLeavesTheHookAfterReleasingIt() {
        val hook = RecordingHook(abortOnCall = 2)
        try {
            NcnnBackend.forward(serialize = true, lowPriority = true, hook = hook) {
                fail("放棄了就不該進前向")
            }
            fail("應拋 NcnnForwardAbortedException")
        } catch (e: NcnnForwardAbortedException) {
            // 預期
        }
        assertEquals(
            listOf("abort? locked=false", "enter locked=false", "abort? locked=true", "exit locked=false"),
            hook.events,
        )
    }

    @Test
    fun unserializedAndUnhookedPathsDoNotEnterTheHook() {
        // 不進鎖（FREE 組）：只在開始前問一次放棄，不進掛鉤
        val hook = RecordingHook()
        assertEquals(7, NcnnBackend.forward(serialize = false, lowPriority = true, hook = hook) { 7 })
        assertEquals(listOf("abort? locked=false"), hook.events)
        // 翻譯（沒有掛鉤）：照樣持鎖跑
        assertTrue(NcnnBackend.forward(serialize = true, lowPriority = false, hook = null) { NcnnBackend.holdsNcnnLock() })
        // 夜讀但沒帶掛鉤：同樣持鎖跑
        assertTrue(NcnnBackend.forward(serialize = true, lowPriority = true, hook = null) { NcnnBackend.holdsNcnnLock() })
    }
}
