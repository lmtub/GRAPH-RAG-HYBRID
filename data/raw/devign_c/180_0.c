static void prop_get_fdt(Object *obj, Visitor *v, const char *name,

                         void *opaque, Error **errp)

{

    sPAPRDRConnector *drc = SPAPR_DR_CONNECTOR(obj);

    Error *err = NULL;

    int fdt_offset_next, fdt_offset, fdt_depth;

    void *fdt;



    if (!drc->fdt) {

        visit_type_null(v, NULL, errp);

        return;

    }



    fdt = drc->fdt;

    fdt_offset = drc->fdt_start_offset;

    fdt_depth = 0;



    do {

        const char *name = NULL;

        const struct fdt_property *prop = NULL;

        int prop_len = 0, name_len = 0;

        uint32_t tag;



        tag = fdt_next_tag(fdt, fdt_offset, &fdt_offset_next);

        switch (tag) {

        case FDT_BEGIN_NODE:

            fdt_depth++;

            name = fdt_get_name(fdt, fdt_offset, &name_len);

            visit_start_struct(v, name, NULL, 0, &err);

            if (err) {

                error_propagate(errp, err);

                return;

            }

            break;

        case FDT_END_NODE:

            /* shouldn't ever see an FDT_END_NODE before FDT_BEGIN_NODE */

            g_assert(fdt_depth > 0);

            visit_check_struct(v, &err);

            visit_end_struct(v);

            if (err) {

                error_propagate(errp, err);

                return;

            }

            fdt_depth--;

            break;

        case FDT_PROP: {

            int i;

            prop = fdt_get_property_by_offset(fdt, fdt_offset, &prop_len);

            name = fdt_string(fdt, fdt32_to_cpu(prop->nameoff));

            visit_start_list(v, name, &err);

            if (err) {

                error_propagate(errp, err);

                return;

            }

            for (i = 0; i < prop_len; i++) {

                visit_type_uint8(v, NULL, (uint8_t *)&prop->data[i], &err);

                if (err) {

                    error_propagate(errp, err);

                    return;

                }

            }

            visit_end_list(v);

            break;

        }

        default:

            error_setg(&error_abort, "device FDT in unexpected state: %d", tag);

        }

        fdt_offset = fdt_offset_next;

    } while (fdt_depth != 0);

}
