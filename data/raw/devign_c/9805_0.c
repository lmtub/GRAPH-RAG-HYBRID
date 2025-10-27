static void flush_queued_work(CPUState *env)

{

    struct qemu_work_item *wi;



    if (!env->queued_work_first)

        return;



    while ((wi = env->queued_work_first)) {

        env->queued_work_first = wi->next;

        wi->func(wi->data);

        wi->done = true;

    }

    env->queued_work_last = NULL;

    qemu_cond_broadcast(&qemu_work_cond);

}
