xmit_seg(E1000State *s)

{

    uint16_t len;

    unsigned int frames = s->tx.tso_frames, css, sofar;

    struct e1000_tx *tp = &s->tx;



    if (tp->props.tse && tp->props.cptse) {

        css = tp->props.ipcss;

        DBGOUT(TXSUM, "frames %d size %d ipcss %d\n",

               frames, tp->size, css);

        if (tp->props.ip) {    /* IPv4 */

            stw_be_p(tp->data+css+2, tp->size - css);

            stw_be_p(tp->data+css+4,

                     lduw_be_p(tp->data + css + 4) + frames);

        } else {         /* IPv6 */

            stw_be_p(tp->data+css+4, tp->size - css);

        }

        css = tp->props.tucss;

        len = tp->size - css;

        DBGOUT(TXSUM, "tcp %d tucss %d len %d\n", tp->props.tcp, css, len);

        if (tp->props.tcp) {

            sofar = frames * tp->props.mss;

            stl_be_p(tp->data+css+4, ldl_be_p(tp->data+css+4)+sofar); /* seq */

            if (tp->props.paylen - sofar > tp->props.mss) {

                tp->data[css + 13] &= ~9;    /* PSH, FIN */

            } else if (frames) {

                e1000x_inc_reg_if_not_full(s->mac_reg, TSCTC);

            }

        } else    /* UDP */

            stw_be_p(tp->data+css+4, len);

        if (tp->props.sum_needed & E1000_TXD_POPTS_TXSM) {

            unsigned int phsum;

            // add pseudo-header length before checksum calculation

            void *sp = tp->data + tp->props.tucso;



            phsum = lduw_be_p(sp) + len;

            phsum = (phsum >> 16) + (phsum & 0xffff);

            stw_be_p(sp, phsum);

        }

        tp->tso_frames++;

    }



    if (tp->props.sum_needed & E1000_TXD_POPTS_TXSM) {

        putsum(tp->data, tp->size, tp->props.tucso,

               tp->props.tucss, tp->props.tucse);

    }

    if (tp->props.sum_needed & E1000_TXD_POPTS_IXSM) {

        putsum(tp->data, tp->size, tp->props.ipcso,

               tp->props.ipcss, tp->props.ipcse);

    }

    if (tp->vlan_needed) {

        memmove(tp->vlan, tp->data, 4);

        memmove(tp->data, tp->data + 4, 8);

        memcpy(tp->data + 8, tp->vlan_header, 4);

        e1000_send_packet(s, tp->vlan, tp->size + 4);

    } else {

        e1000_send_packet(s, tp->data, tp->size);

    }



    e1000x_inc_reg_if_not_full(s->mac_reg, TPT);

    e1000x_grow_8reg_if_not_full(s->mac_reg, TOTL, s->tx.size);

    s->mac_reg[GPTC] = s->mac_reg[TPT];

    s->mac_reg[GOTCL] = s->mac_reg[TOTL];

    s->mac_reg[GOTCH] = s->mac_reg[TOTH];

}
