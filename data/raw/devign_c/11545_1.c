static void qemu_laio_completion_cb(EventNotifier *e)

{

    struct qemu_laio_state *s = container_of(e, struct qemu_laio_state, e);



    while (event_notifier_test_and_clear(&s->e)) {

        struct io_event events[MAX_EVENTS];

        struct timespec ts = { 0 };

        int nevents, i;



        do {

            nevents = io_getevents(s->ctx, MAX_EVENTS, MAX_EVENTS, events, &ts);

        } while (nevents == -EINTR);



        for (i = 0; i < nevents; i++) {

            struct iocb *iocb = events[i].obj;

            struct qemu_laiocb *laiocb =

                    container_of(iocb, struct qemu_laiocb, iocb);



            laiocb->ret = io_event_ret(&events[i]);

            qemu_laio_process_completion(s, laiocb);

        }

    }

}
