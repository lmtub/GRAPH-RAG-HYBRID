static GenericList *next_list(Visitor *v, GenericList **list, size_t size)

{

    StringOutputVisitor *sov = to_sov(v);

    GenericList *ret = NULL;

    if (*list) {

        if (sov->head) {

            ret = *list;

        } else {

            ret = (*list)->next;

        }



        if (sov->head) {

            if (ret && ret->next == NULL) {

                sov->list_mode = LM_NONE;

            }

            sov->head = false;

        } else {

            if (ret && ret->next == NULL) {

                sov->list_mode = LM_END;

            }

        }

    }



    return ret;

}
