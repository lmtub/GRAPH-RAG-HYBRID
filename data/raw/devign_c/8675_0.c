static void openpic_save(QEMUFile* f, void *opaque)

{

    OpenPICState *opp = (OpenPICState *)opaque;

    unsigned int i;



    qemu_put_be32s(f, &opp->glbc);

    qemu_put_be32s(f, &opp->veni);

    qemu_put_be32s(f, &opp->pint);

    qemu_put_be32s(f, &opp->spve);

    qemu_put_be32s(f, &opp->tifr);



    for (i = 0; i < opp->max_irq; i++) {

        qemu_put_be32s(f, &opp->src[i].ipvp);

        qemu_put_be32s(f, &opp->src[i].ide);

        qemu_put_sbe32s(f, &opp->src[i].last_cpu);

        qemu_put_sbe32s(f, &opp->src[i].pending);

    }



    qemu_put_be32s(f, &opp->nb_cpus);



    for (i = 0; i < opp->nb_cpus; i++) {

        qemu_put_be32s(f, &opp->dst[i].pctp);

        qemu_put_be32s(f, &opp->dst[i].pcsr);

        openpic_save_IRQ_queue(f, &opp->dst[i].raised);

        openpic_save_IRQ_queue(f, &opp->dst[i].servicing);

    }



    for (i = 0; i < MAX_TMR; i++) {

        qemu_put_be32s(f, &opp->timers[i].ticc);

        qemu_put_be32s(f, &opp->timers[i].tibc);

    }

}
