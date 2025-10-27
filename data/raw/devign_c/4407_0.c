static void ioq_submit(LinuxAioState *s)

{

    int ret, len;

    struct qemu_laiocb *aiocb;

    struct iocb *iocbs[MAX_QUEUED_IO];

    QSIMPLEQ_HEAD(, qemu_laiocb) completed;



    do {

        len = 0;

        QSIMPLEQ_FOREACH(aiocb, &s->io_q.pending, next) {

            iocbs[len++] = &aiocb->iocb;

            if (len == MAX_QUEUED_IO) {

                break;

            }

        }



        ret = io_submit(s->ctx, len, iocbs);

        if (ret == -EAGAIN) {

            break;

        }

        if (ret < 0) {

            abort();

        }



        s->io_q.n -= ret;

        aiocb = container_of(iocbs[ret - 1], struct qemu_laiocb, iocb);

        QSIMPLEQ_SPLIT_AFTER(&s->io_q.pending, aiocb, next, &completed);

    } while (ret == len && !QSIMPLEQ_EMPTY(&s->io_q.pending));

    s->io_q.blocked = (s->io_q.n > 0);

}
