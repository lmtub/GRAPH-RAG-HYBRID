static void acpi_table_install(const char unsigned *blob, size_t bloblen,

                               bool has_header,

                               const struct AcpiTableOptions *hdrs,

                               Error **errp)

{

    size_t body_start;

    const char unsigned *hdr_src;

    size_t body_size, acpi_payload_size;

    struct acpi_table_header *ext_hdr;

    unsigned changed_fields;



    /* Calculate where the ACPI table body starts within the blob, plus where

     * to copy the ACPI table header from.

     */

    if (has_header) {

        /*   _length             | ACPI header in blob | blob body

         *   ^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^

         *   ACPI_TABLE_PFX_SIZE     sizeof dfl_hdr      body_size

         *                           == body_start

         *

         *                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

         *                           acpi_payload_size == bloblen

         */

        body_start = sizeof dfl_hdr;



        if (bloblen < body_start) {

            error_setg(errp, "ACPI table claiming to have header is too "

                       "short, available: %zu, expected: %zu", bloblen,

                       body_start);

            return;

        }

        hdr_src = blob;

    } else {

        /*   _length             | ACPI header in template | blob body

         *   ^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^

         *   ACPI_TABLE_PFX_SIZE       sizeof dfl_hdr        body_size

         *                                                   == bloblen

         *

         *                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

         *                                  acpi_payload_size

         */

        body_start = 0;

        hdr_src = dfl_hdr;

    }

    body_size = bloblen - body_start;

    acpi_payload_size = sizeof dfl_hdr + body_size;



    if (acpi_payload_size > UINT16_MAX) {

        error_setg(errp, "ACPI table too big, requested: %zu, max: %u",

                   acpi_payload_size, (unsigned)UINT16_MAX);

        return;

    }



    /* We won't fail from here on. Initialize / extend the globals. */

    if (acpi_tables == NULL) {

        acpi_tables_len = sizeof(uint16_t);

        acpi_tables = g_malloc0(acpi_tables_len);

    }



    acpi_tables = g_realloc(acpi_tables, acpi_tables_len +

                                         ACPI_TABLE_PFX_SIZE +

                                         sizeof dfl_hdr + body_size);



    ext_hdr = (struct acpi_table_header *)(acpi_tables + acpi_tables_len);

    acpi_tables_len += ACPI_TABLE_PFX_SIZE;



    memcpy(acpi_tables + acpi_tables_len, hdr_src, sizeof dfl_hdr);

    acpi_tables_len += sizeof dfl_hdr;



    if (blob != NULL) {

        memcpy(acpi_tables + acpi_tables_len, blob + body_start, body_size);

        acpi_tables_len += body_size;

    }



    /* increase number of tables */

    stw_le_p(acpi_tables, lduw_le_p(acpi_tables) + 1u);



    /* Update the header fields. The strings need not be NUL-terminated. */

    changed_fields = 0;

    ext_hdr->_length = cpu_to_le16(acpi_payload_size);



    if (hdrs->has_sig) {

        strncpy(ext_hdr->sig, hdrs->sig, sizeof ext_hdr->sig);

        ++changed_fields;

    }



    if (has_header && le32_to_cpu(ext_hdr->length) != acpi_payload_size) {

        fprintf(stderr,

                "warning: ACPI table has wrong length, header says "

                "%" PRIu32 ", actual size %zu bytes\n",

                le32_to_cpu(ext_hdr->length), acpi_payload_size);

    }

    ext_hdr->length = cpu_to_le32(acpi_payload_size);



    if (hdrs->has_rev) {

        ext_hdr->revision = hdrs->rev;

        ++changed_fields;

    }



    ext_hdr->checksum = 0;



    if (hdrs->has_oem_id) {

        strncpy(ext_hdr->oem_id, hdrs->oem_id, sizeof ext_hdr->oem_id);

        ++changed_fields;

    }

    if (hdrs->has_oem_table_id) {

        strncpy(ext_hdr->oem_table_id, hdrs->oem_table_id,

                sizeof ext_hdr->oem_table_id);

        ++changed_fields;

    }

    if (hdrs->has_oem_rev) {

        ext_hdr->oem_revision = cpu_to_le32(hdrs->oem_rev);

        ++changed_fields;

    }

    if (hdrs->has_asl_compiler_id) {

        strncpy(ext_hdr->asl_compiler_id, hdrs->asl_compiler_id,

                sizeof ext_hdr->asl_compiler_id);

        ++changed_fields;

    }

    if (hdrs->has_asl_compiler_rev) {

        ext_hdr->asl_compiler_revision = cpu_to_le32(hdrs->asl_compiler_rev);

        ++changed_fields;

    }



    if (!has_header && changed_fields == 0) {

        warn_report("ACPI table: no headers are specified");

    }



    /* recalculate checksum */

    ext_hdr->checksum = acpi_checksum((const char unsigned *)ext_hdr +

                                      ACPI_TABLE_PFX_SIZE, acpi_payload_size);

}
