static inline void array_free(array_t* array)

{

    if(array->pointer)

        free(array->pointer);

    array->size=array->next=0;

}
