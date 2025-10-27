static int vmstate_size(void *opaque, VMStateField *field)

{

    int size = field->size;



    if (field->flags & VMS_VBUFFER) {

        size = *(int32_t *)(opaque+field->size_offset);

        if (field->flags & VMS_MULTIPLY) {

            size *= field->size;

        }

    }



    return size;

}
