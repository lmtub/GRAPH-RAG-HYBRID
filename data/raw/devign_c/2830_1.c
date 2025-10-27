static void sdp_service_record_build(struct sdp_service_record_s *record,

                struct sdp_def_service_s *def, int handle)

{

    int len = 0;

    uint8_t *data;

    int *uuid;



    record->uuids = 0;

    while (def->attributes[record->attributes].data.type) {

        len += 3;

        len += sdp_attr_max_size(&def->attributes[record->attributes ++].data,

                        &record->uuids);

    }

    record->uuids = pow2ceil(record->uuids);

    record->attribute_list =

            g_malloc0(record->attributes * sizeof(*record->attribute_list));

    record->uuid =

            g_malloc0(record->uuids * sizeof(*record->uuid));

    data = g_malloc(len);



    record->attributes = 0;

    uuid = record->uuid;

    while (def->attributes[record->attributes].data.type) {

        record->attribute_list[record->attributes].pair = data;



        len = 0;

        data[len ++] = SDP_DTYPE_UINT | SDP_DSIZE_2;

        data[len ++] = def->attributes[record->attributes].id >> 8;

        data[len ++] = def->attributes[record->attributes].id & 0xff;

        len += sdp_attr_write(data + len,

                        &def->attributes[record->attributes].data, &uuid);



        /* Special case: assign a ServiceRecordHandle in sequence */

        if (def->attributes[record->attributes].id == SDP_ATTR_RECORD_HANDLE)

            def->attributes[record->attributes].data.value.uint = handle;

        /* Note: we could also assign a ServiceDescription based on

         * sdp->device.device->lmp_name.  */



        record->attribute_list[record->attributes ++].len = len;

        data += len;

    }



    /* Sort the attribute list by the AttributeID */

    qsort(record->attribute_list, record->attributes,

                    sizeof(*record->attribute_list),

                    (void *) sdp_attributeid_compare);

    /* Sort the searchable UUIDs list for bisection */

    qsort(record->uuid, record->uuids,

                    sizeof(*record->uuid),

                    (void *) sdp_uuid_compare);

}
