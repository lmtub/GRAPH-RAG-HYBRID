void qsb_free(QEMUSizedBuffer *qsb)

{

    size_t i;



    if (!qsb) {

        return;

    }



    for (i = 0; i < qsb->n_iov; i++) {

        g_free(qsb->iov[i].iov_base);

    }

    g_free(qsb->iov);

    g_free(qsb);

}
