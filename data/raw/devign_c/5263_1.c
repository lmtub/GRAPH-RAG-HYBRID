static inline void IRQ_setbit(IRQ_queue_t *q, int n_IRQ)

{

    q->pending++;

    set_bit(q->queue, n_IRQ);

}
