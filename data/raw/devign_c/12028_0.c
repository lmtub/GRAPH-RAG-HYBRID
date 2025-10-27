static void ptimer_trigger(ptimer_state *s)

{

    if (s->bh) {

        qemu_bh_schedule(s->bh);

    }

}
