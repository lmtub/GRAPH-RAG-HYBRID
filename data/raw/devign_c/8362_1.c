static void cpu_4xx_wdt_cb (void *opaque)

{

    PowerPCCPU *cpu;

    CPUPPCState *env;

    ppc_tb_t *tb_env;

    ppc40x_timer_t *ppc40x_timer;

    uint64_t now, next;



    env = opaque;

    cpu = ppc_env_get_cpu(env);

    tb_env = env->tb_env;

    ppc40x_timer = tb_env->opaque;

    now = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);

    switch ((env->spr[SPR_40x_TCR] >> 30) & 0x3) {

    case 0:

        next = 1 << 17;

        break;

    case 1:

        next = 1 << 21;

        break;

    case 2:

        next = 1 << 25;

        break;

    case 3:

        next = 1 << 29;

        break;

    default:

        /* Cannot occur, but makes gcc happy */

        return;

    }

    next = now + muldiv64(next, get_ticks_per_sec(), tb_env->decr_freq);

    if (next == now)

        next++;

    LOG_TB("%s: TCR " TARGET_FMT_lx " TSR " TARGET_FMT_lx "\n", __func__,

           env->spr[SPR_40x_TCR], env->spr[SPR_40x_TSR]);

    switch ((env->spr[SPR_40x_TSR] >> 30) & 0x3) {

    case 0x0:

    case 0x1:

        timer_mod(ppc40x_timer->wdt_timer, next);

        ppc40x_timer->wdt_next = next;

        env->spr[SPR_40x_TSR] |= 1 << 31;

        break;

    case 0x2:

        timer_mod(ppc40x_timer->wdt_timer, next);

        ppc40x_timer->wdt_next = next;

        env->spr[SPR_40x_TSR] |= 1 << 30;

        if ((env->spr[SPR_40x_TCR] >> 27) & 0x1) {

            ppc_set_irq(cpu, PPC_INTERRUPT_WDT, 1);

        }

        break;

    case 0x3:

        env->spr[SPR_40x_TSR] &= ~0x30000000;

        env->spr[SPR_40x_TSR] |= env->spr[SPR_40x_TCR] & 0x30000000;

        switch ((env->spr[SPR_40x_TCR] >> 28) & 0x3) {

        case 0x0:

            /* No reset */

            break;

        case 0x1: /* Core reset */

            ppc40x_core_reset(cpu);

            break;

        case 0x2: /* Chip reset */

            ppc40x_chip_reset(cpu);

            break;

        case 0x3: /* System reset */

            ppc40x_system_reset(cpu);

            break;

        }

    }

}
