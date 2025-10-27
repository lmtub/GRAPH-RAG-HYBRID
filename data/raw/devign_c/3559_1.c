static inline void write_mem(IVState *s, uint64_t off,

                             const void *buf, size_t len)

{

    QTestState *qtest = global_qtest;



    global_qtest = s->qtest;

    qpci_memwrite(s->dev, s->mem_base + off, buf, len);

    global_qtest = qtest;

}
