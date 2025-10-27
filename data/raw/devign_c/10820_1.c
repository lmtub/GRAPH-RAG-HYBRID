static void *vmstate_base_addr(void *opaque, VMStateField *field, bool alloc)

{

    void *base_addr = opaque + field->offset;



    if (field->flags & VMS_POINTER) {

        if (alloc && (field->flags & VMS_ALLOC)) {

            gsize size = 0;

            if (field->flags & VMS_VBUFFER) {

                size = vmstate_size(opaque, field);

            } else {

                int n_elems = vmstate_n_elems(opaque, field);

                if (n_elems) {

                    size = n_elems * field->size;

                }

            }

            if (size) {

                *((void **)base_addr + field->start) = g_malloc(size);

            }

        }

        base_addr = *(void **)base_addr + field->start;

    }



    return base_addr;

}
