void virtqueue_get_avail_bytes(VirtQueue *vq, unsigned int *in_bytes,

                               unsigned int *out_bytes)

{

    unsigned int idx;

    unsigned int total_bufs, in_total, out_total;



    idx = vq->last_avail_idx;



    total_bufs = in_total = out_total = 0;

    while (virtqueue_num_heads(vq, idx)) {

        unsigned int max, num_bufs, indirect = 0;

        hwaddr desc_pa;

        int i;



        max = vq->vring.num;

        num_bufs = total_bufs;

        i = virtqueue_get_head(vq, idx++);

        desc_pa = vq->vring.desc;



        if (vring_desc_flags(desc_pa, i) & VRING_DESC_F_INDIRECT) {

            if (vring_desc_len(desc_pa, i) % sizeof(VRingDesc)) {

                error_report("Invalid size for indirect buffer table");

                exit(1);

            }



            /* If we've got too many, that implies a descriptor loop. */

            if (num_bufs >= max) {

                error_report("Looped descriptor");

                exit(1);

            }



            /* loop over the indirect descriptor table */

            indirect = 1;

            max = vring_desc_len(desc_pa, i) / sizeof(VRingDesc);

            num_bufs = i = 0;

            desc_pa = vring_desc_addr(desc_pa, i);

        }



        do {

            /* If we've got too many, that implies a descriptor loop. */

            if (++num_bufs > max) {

                error_report("Looped descriptor");

                exit(1);

            }



            if (vring_desc_flags(desc_pa, i) & VRING_DESC_F_WRITE) {

                in_total += vring_desc_len(desc_pa, i);

            } else {

                out_total += vring_desc_len(desc_pa, i);

            }

        } while ((i = virtqueue_next_desc(desc_pa, i, max)) != max);



        if (!indirect)

            total_bufs = num_bufs;

        else

            total_bufs++;

    }

    if (in_bytes) {

        *in_bytes = in_total;

    }

    if (out_bytes) {

        *out_bytes = out_total;

    }

}
