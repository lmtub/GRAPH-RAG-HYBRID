static void parse_header_digest(struct iscsi_context *iscsi, const char *target)

{

    QemuOptsList *list;

    QemuOpts *opts;

    const char *digest = NULL;



    list = qemu_find_opts("iscsi");

    if (!list) {

        return;

    }



    opts = qemu_opts_find(list, target);

    if (opts == NULL) {

        opts = QTAILQ_FIRST(&list->head);

        if (!opts) {

            return;

        }

    }



    digest = qemu_opt_get(opts, "header-digest");

    if (!digest) {

        return;

    }



    if (!strcmp(digest, "CRC32C")) {

        iscsi_set_header_digest(iscsi, ISCSI_HEADER_DIGEST_CRC32C);

    } else if (!strcmp(digest, "NONE")) {

        iscsi_set_header_digest(iscsi, ISCSI_HEADER_DIGEST_NONE);

    } else if (!strcmp(digest, "CRC32C-NONE")) {

        iscsi_set_header_digest(iscsi, ISCSI_HEADER_DIGEST_CRC32C_NONE);

    } else if (!strcmp(digest, "NONE-CRC32C")) {

        iscsi_set_header_digest(iscsi, ISCSI_HEADER_DIGEST_NONE_CRC32C);

    } else {

        error_report("Invalid header-digest setting : %s", digest);

    }

}
