static void pfpu_start(MilkymistPFPUState *s)

{

    int x, y;

    int i;



    for (y = 0; y <= s->regs[R_VMESHLAST]; y++) {

        for (x = 0; x <= s->regs[R_HMESHLAST]; x++) {

            D_EXEC(qemu_log("\nprocessing x=%d y=%d\n", x, y));



            /* set current position */

            s->gp_regs[GPR_X] = x;

            s->gp_regs[GPR_Y] = y;



            /* run microcode on this position */

            i = 0;

            while (pfpu_decode_insn(s)) {

                /* decode at most MICROCODE_WORDS instructions */

                if (i++ >= MICROCODE_WORDS) {

                    error_report("milkymist_pfpu: too many instructions "

                            "executed in microcode. No VECTOUT?");

                    break;

                }

            }



            /* reset pc for next run */

            s->regs[R_PC] = 0;

        }

    }



    s->regs[R_VERTICES] = x * y;



    trace_milkymist_pfpu_pulse_irq();

    qemu_irq_pulse(s->irq);

}
