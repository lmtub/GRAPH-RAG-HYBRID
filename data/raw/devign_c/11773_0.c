static void ide_atapi_cmd_ok(IDEState *s)

{

    s->error = 0;

    s->status = READY_STAT;

    s->nsector = (s->nsector & ~7) | ATAPI_INT_REASON_IO | ATAPI_INT_REASON_CD;

    ide_set_irq(s);

}
