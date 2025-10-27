QDict *qtest_qmpv(QTestState *s, const char *fmt, va_list ap)

{

    /* Send QMP request */

    socket_sendf(s->qmp_fd, fmt, ap);



    /* Receive reply */

    return qtest_qmp_receive(s);

}
