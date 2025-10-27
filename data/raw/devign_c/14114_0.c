static inline void IRQ_resetbit(IRQQueue *q, int n_IRQ)

{

    reset_bit(q->queue, n_IRQ);

}
