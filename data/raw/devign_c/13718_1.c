int main(void){

    int i,k;

    AVTreeNode *root= NULL, *node=NULL;



    for(i=0; i<10000; i++){

        int j= (random()%86294);

        if(check(root) > 999){

            av_log(NULL, AV_LOG_ERROR, "FATAL error %d\n", i);

        print(root, 0);

            return -1;

        }

        av_log(NULL, AV_LOG_ERROR, "inserting %4d\n", j);

        if(!node)

            node= av_mallocz(av_tree_node_size);

        av_tree_insert(&root, (void*)(j+1), cmp, &node);



        j= (random()%86294);

        k= av_tree_find(root, (void*)(j+1), cmp, NULL);

        if(k){

            AVTreeNode *node2=NULL;

            av_log(NULL, AV_LOG_ERROR, "removing %4d\n", j);

            av_tree_insert(&root, (void*)(j+1), cmp, &node2);

            k= av_tree_find(root, (void*)(j+1), cmp, NULL);

            if(k)

                av_log(NULL, AV_LOG_ERROR, "removial failure %d\n", i);

        }

    }

    return 0;

}
