static int buf_close(void *opaque)

{

    QEMUBuffer *s = opaque;



    qsb_free(s->qsb);



    g_free(s);



    return 0;

}
