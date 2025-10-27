static void stm32f2xx_timer_write(void *opaque, hwaddr offset,

                        uint64_t val64, unsigned size)

{

    STM32F2XXTimerState *s = opaque;

    uint32_t value = val64;

    int64_t now = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);

    uint32_t timer_val = 0;



    DB_PRINT("Write 0x%x, 0x%"HWADDR_PRIx"\n", value, offset);



    switch (offset) {

    case TIM_CR1:

        s->tim_cr1 = value;

        return;

    case TIM_CR2:

        s->tim_cr2 = value;

        return;

    case TIM_SMCR:

        s->tim_smcr = value;

        return;

    case TIM_DIER:

        s->tim_dier = value;

        return;

    case TIM_SR:

        /* This is set by hardware and cleared by software */

        s->tim_sr &= value;

        return;

    case TIM_EGR:

        s->tim_egr = value;

        if (s->tim_egr & TIM_EGR_UG) {

            timer_val = 0;

            break;

        }

        return;

    case TIM_CCMR1:

        s->tim_ccmr1 = value;

        return;

    case TIM_CCMR2:

        s->tim_ccmr2 = value;

        return;

    case TIM_CCER:

        s->tim_ccer = value;

        return;

    case TIM_PSC:

        timer_val = stm32f2xx_ns_to_ticks(s, now) - s->tick_offset;

        s->tim_psc = value;

        value = timer_val;

        break;

    case TIM_CNT:

        timer_val = value;

        break;

    case TIM_ARR:

        s->tim_arr = value;

        stm32f2xx_timer_set_alarm(s, now);

        return;

    case TIM_CCR1:

        s->tim_ccr1 = value;

        return;

    case TIM_CCR2:

        s->tim_ccr2 = value;

        return;

    case TIM_CCR3:

        s->tim_ccr3 = value;

        return;

    case TIM_CCR4:

        s->tim_ccr4 = value;

        return;

    case TIM_DCR:

        s->tim_dcr = value;

        return;

    case TIM_DMAR:

        s->tim_dmar = value;

        return;

    case TIM_OR:

        s->tim_or = value;

        return;

    default:

        qemu_log_mask(LOG_GUEST_ERROR,

                      "%s: Bad offset 0x%"HWADDR_PRIx"\n", __func__, offset);

        return;

    }



    /* This means that a register write has affected the timer in a way that

     * requires a refresh of both tick_offset and the alarm.

     */

    s->tick_offset = stm32f2xx_ns_to_ticks(s, now) - timer_val;

    stm32f2xx_timer_set_alarm(s, now);

}
