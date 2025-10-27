static void openpic_load_IRQ_queue(QEMUFile* f, IRQ_queue_t *q)

{

    unsigned int i;



    for (i = 0; i < BF_WIDTH(MAX_IRQ); i++)

        qemu_get_be32s(f, &q->queue[i]);



    qemu_get_sbe32s(f, &q->next);

    qemu_get_sbe32s(f, &q->priority);

}
