static void eth_send(mv88w8618_eth_state *s, int queue_index)

{

    uint32_t desc_addr = s->tx_queue[queue_index];

    mv88w8618_tx_desc desc;

    uint8_t buf[2048];

    int len;





    do {

        eth_tx_desc_get(desc_addr, &desc);

        if (desc.cmdstat & MP_ETH_TX_OWN) {

            len = desc.bytes;

            if (len < 2048) {

                cpu_physical_memory_read(desc.buffer, buf, len);

                qemu_send_packet(s->vc, buf, len);

            }

            desc.cmdstat &= ~MP_ETH_TX_OWN;

            s->icr |= 1 << (MP_ETH_IRQ_TXLO_BIT - queue_index);

            eth_tx_desc_put(desc_addr, &desc);

        }

        desc_addr = desc.next;

    } while (desc_addr != s->tx_queue[queue_index]);

}
