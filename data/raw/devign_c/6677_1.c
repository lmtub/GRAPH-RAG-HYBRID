QemuOptsList *qemu_opts_append(QemuOptsList *dst,

                               QemuOptsList *list)

{

    size_t num_opts, num_dst_opts;

    QemuOptDesc *desc;

    bool need_init = false;



    if (!list) {

        return dst;

    }



    /* If dst is NULL, after realloc, some area of dst should be initialized

     * before adding options to it.

     */

    if (!dst) {

        need_init = true;

    }



    num_opts = count_opts_list(dst);

    num_dst_opts = num_opts;

    num_opts += count_opts_list(list);

    dst = g_realloc(dst, sizeof(QemuOptsList) +

                    (num_opts + 1) * sizeof(QemuOptDesc));

    if (need_init) {

        dst->name = NULL;

        dst->implied_opt_name = NULL;

        QTAILQ_INIT(&dst->head);

        dst->merge_lists = false;

    }

    dst->desc[num_dst_opts].name = NULL;



    /* append list->desc to dst->desc */

    if (list) {

        desc = list->desc;

        while (desc && desc->name) {

            if (find_desc_by_name(dst->desc, desc->name) == NULL) {

                dst->desc[num_dst_opts++] = *desc;

                dst->desc[num_dst_opts].name = NULL;

            }

            desc++;

        }

    }



    return dst;

}
