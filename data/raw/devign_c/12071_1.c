static MegasasCmd *megasas_enqueue_frame(MegasasState *s,

    hwaddr frame, uint64_t context, int count)

{

    PCIDevice *pcid = PCI_DEVICE(s);

    MegasasCmd *cmd = NULL;

    int frame_size = MFI_FRAME_SIZE * 16;

    hwaddr frame_size_p = frame_size;



    cmd = megasas_next_frame(s, frame);

    /* All frames busy */

    if (!cmd) {

        return NULL;

    }

    if (!cmd->pa) {

        cmd->pa = frame;

        /* Map all possible frames */

        cmd->frame = pci_dma_map(pcid, frame, &frame_size_p, 0);

        if (frame_size_p != frame_size) {

            trace_megasas_qf_map_failed(cmd->index, (unsigned long)frame);

            if (cmd->frame) {

                pci_dma_unmap(pcid, cmd->frame, frame_size_p, 0, 0);

                cmd->frame = NULL;

                cmd->pa = 0;

            }

            s->event_count++;

            return NULL;

        }

        cmd->pa_size = frame_size_p;

        cmd->context = context;

        if (!megasas_use_queue64(s)) {

            cmd->context &= (uint64_t)0xFFFFFFFF;

        }

    }

    cmd->count = count;

    s->busy++;



    if (s->consumer_pa) {

        s->reply_queue_tail = ldl_le_phys(&address_space_memory,

                                          s->consumer_pa);

    }

    trace_megasas_qf_enqueue(cmd->index, cmd->count, cmd->context,

                             s->reply_queue_head, s->reply_queue_tail, s->busy);



    return cmd;

}
