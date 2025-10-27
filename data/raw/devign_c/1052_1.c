xmit_seg(E1000State *s)

{

    uint16_t len, *sp;

    unsigned int frames = s->tx.tso_frames, css, sofar, n;

    struct e1000_tx *tp = &s->tx;



    if (tp->tse && tp->cptse) {

        css = tp->ipcss;

        DBGOUT(TXSUM, "frames %d size %d ipcss %d\n",

               frames, tp->size, css);

        if (tp->ip) {		// IPv4

            cpu_to_be16wu((uint16_t *)(tp->data+css+2),

                          tp->size - css);

            cpu_to_be16wu((uint16_t *)(tp->data+css+4),

                          be16_to_cpup((uint16_t *)(tp->data+css+4))+frames);

        } else			// IPv6

            cpu_to_be16wu((uint16_t *)(tp->data+css+4),

                          tp->size - css);

        css = tp->tucss;

        len = tp->size - css;

        DBGOUT(TXSUM, "tcp %d tucss %d len %d\n", tp->tcp, css, len);

        if (tp->tcp) {

            sofar = frames * tp->mss;

            cpu_to_be32wu((uint32_t *)(tp->data+css+4),	// seq

                be32_to_cpupu((uint32_t *)(tp->data+css+4))+sofar);

            if (tp->paylen - sofar > tp->mss)

                tp->data[css + 13] &= ~9;		// PSH, FIN

        } else	// UDP

            cpu_to_be16wu((uint16_t *)(tp->data+css+4), len);

        if (tp->sum_needed & E1000_TXD_POPTS_TXSM) {

            // add pseudo-header length before checksum calculation

            sp = (uint16_t *)(tp->data + tp->tucso);

            cpu_to_be16wu(sp, be16_to_cpup(sp) + len);

        }

        tp->tso_frames++;

    }



    if (tp->sum_needed & E1000_TXD_POPTS_TXSM)

        putsum(tp->data, tp->size, tp->tucso, tp->tucss, tp->tucse);

    if (tp->sum_needed & E1000_TXD_POPTS_IXSM)

        putsum(tp->data, tp->size, tp->ipcso, tp->ipcss, tp->ipcse);

    if (tp->vlan_needed) {

        memmove(tp->vlan, tp->data, 4);

        memmove(tp->data, tp->data + 4, 8);

        memcpy(tp->data + 8, tp->vlan_header, 4);

        qemu_send_packet(&s->nic->nc, tp->vlan, tp->size + 4);

    } else

        qemu_send_packet(&s->nic->nc, tp->data, tp->size);

    s->mac_reg[TPT]++;

    s->mac_reg[GPTC]++;

    n = s->mac_reg[TOTL];

    if ((s->mac_reg[TOTL] += s->tx.size) < n)

        s->mac_reg[TOTH]++;

}
