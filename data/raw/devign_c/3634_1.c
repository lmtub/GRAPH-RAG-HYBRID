static void nvme_process_db(NvmeCtrl *n, hwaddr addr, int val)

{

    uint32_t qid;



    if (addr & ((1 << 2) - 1)) {

        return;

    }



    if (((addr - 0x1000) >> 2) & 1) {

        uint16_t new_head = val & 0xffff;

        int start_sqs;

        NvmeCQueue *cq;



        qid = (addr - (0x1000 + (1 << 2))) >> 3;

        if (nvme_check_cqid(n, qid)) {

            return;

        }



        cq = n->cq[qid];

        if (new_head >= cq->size) {

            return;

        }



        start_sqs = nvme_cq_full(cq) ? 1 : 0;

        cq->head = new_head;

        if (start_sqs) {

            NvmeSQueue *sq;

            QTAILQ_FOREACH(sq, &cq->sq_list, entry) {

                timer_mod(sq->timer, qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) + 500);

            }

            timer_mod(cq->timer, qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) + 500);

        }



        if (cq->tail != cq->head) {

            nvme_isr_notify(n, cq);

        }

    } else {

        uint16_t new_tail = val & 0xffff;

        NvmeSQueue *sq;



        qid = (addr - 0x1000) >> 3;

        if (nvme_check_sqid(n, qid)) {

            return;

        }



        sq = n->sq[qid];

        if (new_tail >= sq->size) {

            return;

        }



        sq->tail = new_tail;

        timer_mod(sq->timer, qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) + 500);

    }

}
