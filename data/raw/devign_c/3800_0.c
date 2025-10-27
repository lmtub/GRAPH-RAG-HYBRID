void qemu_bh_schedule_idle(QEMUBH *bh)

{

    if (bh->scheduled)

        return;

    bh->scheduled = 1;

    bh->idle = 1;

}
