static void ne2000_receive(void *opaque, const uint8_t *buf, int size)

{

    NE2000State *s = opaque;

    uint8_t *p;

    int total_len, next, avail, len, index, mcast_idx;

    uint8_t buf1[60];

    static const uint8_t broadcast_macaddr[6] = 

        { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };

    

#if defined(DEBUG_NE2000)

    printf("NE2000: received len=%d\n", size);

#endif



    if (!ne2000_can_receive(s))

        return;

    

    /* XXX: check this */

    if (s->rxcr & 0x10) {

        /* promiscuous: receive all */

    } else {

        if (!memcmp(buf,  broadcast_macaddr, 6)) {

            /* broadcast address */

            if (!(s->rxcr & 0x04))

                return;

        } else if (buf[0] & 0x01) {

            /* multicast */

            if (!(s->rxcr & 0x08))

                return;

            mcast_idx = compute_mcast_idx(buf);

            if (!(s->mult[mcast_idx >> 3] & (1 << (mcast_idx & 7))))

                return;

        } else if (s->mem[0] == buf[0] &&

                   s->mem[2] == buf[1] &&                   

                   s->mem[4] == buf[2] &&            

                   s->mem[6] == buf[3] &&            

                   s->mem[8] == buf[4] &&            

                   s->mem[10] == buf[5]) {

            /* match */

        } else {

            return;

        }

    }





    /* if too small buffer, then expand it */

    if (size < MIN_BUF_SIZE) {

        memcpy(buf1, buf, size);

        memset(buf1 + size, 0, MIN_BUF_SIZE - size);

        buf = buf1;

        size = MIN_BUF_SIZE;

    }



    index = s->curpag << 8;

    /* 4 bytes for header */

    total_len = size + 4;

    /* address for next packet (4 bytes for CRC) */

    next = index + ((total_len + 4 + 255) & ~0xff);

    if (next >= s->stop)

        next -= (s->stop - s->start);

    /* prepare packet header */

    p = s->mem + index;

    s->rsr = ENRSR_RXOK; /* receive status */

    /* XXX: check this */

    if (buf[0] & 0x01)

        s->rsr |= ENRSR_PHY;

    p[0] = s->rsr;

    p[1] = next >> 8;

    p[2] = total_len;

    p[3] = total_len >> 8;

    index += 4;



    /* write packet data */

    while (size > 0) {

        avail = s->stop - index;

        len = size;

        if (len > avail)

            len = avail;

        memcpy(s->mem + index, buf, len);

        buf += len;

        index += len;

        if (index == s->stop)

            index = s->start;

        size -= len;

    }

    s->curpag = next >> 8;



    /* now we can signal we have receive something */

    s->isr |= ENISR_RX;

    ne2000_update_irq(s);

}
