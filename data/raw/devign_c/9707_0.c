static void flush_queued_work(CPUState *cpu)

{

    struct qemu_work_item *wi;



    if (cpu->queued_work_first == NULL) {

        return;

    }



    while ((wi = cpu->queued_work_first)) {

        cpu->queued_work_first = wi->next;

        wi->func(wi->data);

        wi->done = true;

        if (wi->free) {

            g_free(wi);

        }

    }

    cpu->queued_work_last = NULL;

    qemu_cond_broadcast(&qemu_work_cond);

}
