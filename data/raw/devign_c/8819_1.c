static inline void out_reg(IVState *s, enum Reg reg, unsigned v)

{

    const char *name = reg2str(reg);

    QTestState *qtest = global_qtest;



    global_qtest = s->qtest;

    g_test_message("%x -> *%s\n", v, name);

    qpci_io_writel(s->dev, s->reg_base + reg, v);

    global_qtest = qtest;

}
