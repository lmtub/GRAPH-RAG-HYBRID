static void eth_receive(void *opaque, const uint8_t *buf, size_t size)

{

    mv88w8618_eth_state *s = opaque;

    uint32_t desc_addr;

    mv88w8618_rx_desc desc;

    int i;



    for (i = 0; i < 4; i++) {

        desc_addr = s->cur_rx[i];

        if (!desc_addr)

            continue;

        do {

            eth_rx_desc_get(desc_addr, &desc);

            if ((desc.cmdstat & MP_ETH_RX_OWN) && desc.buffer_size >= size) {

                cpu_physical_memory_write(desc.buffer + s->vlan_header,

                                          buf, size);

                desc.bytes = size + s->vlan_header;

                desc.cmdstat &= ~MP_ETH_RX_OWN;

                s->cur_rx[i] = desc.next;



                s->icr |= MP_ETH_IRQ_RX;

                if (s->icr & s->imr)

                    qemu_irq_raise(s->irq);

                eth_rx_desc_put(desc_addr, &desc);

                return;

            }

            desc_addr = desc.next;

        } while (desc_addr != s->rx_queue[i]);

    }

}
