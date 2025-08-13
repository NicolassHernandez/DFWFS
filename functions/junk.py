# nZ24
if wfs.routine == 'nsold_p_nZM24':# 2
    routine = [
        #{'tag':'D-MVM','checkpoint':'./MODELS/old/n0_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_2$D-MVM','checkpoint':'./MODELS/old/n2_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'$\Lambda_3$D-MVM','checkpoint':'./MODELS/old/n3_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_4$D-MVM','checkpoint':'./MODELS/old/n4_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_5$D-MVM','checkpoint':'./MODELS/old/n5_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'}
        ] 
if wfs.routine == 'nsold_nn_nZM24':# 1
    routine = [
        {'tag':'$\Lambda_2$-NN','checkpoint':'./MODELS/old/n2_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'$\Lambda_3$-NN','checkpoint':'./MODELS/old/n3_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_4$-NN','checkpoint':'./MODELS/old/n4_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_5$-NN','checkpoint':'./MODELS/old/n5_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'}
        ] 
if wfs.routine == 'nsold_pnn_nZM24':# 2
    routine = [
        #{'tag':'D-NN','checkpoint':'./MODELS/old/n0_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_2$D-NN','checkpoint':'./MODELS/old/n2_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},   
        {'tag':'$\Lambda_3$D-NN','checkpoint':'./MODELS/old/n3_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_4$D-NN','checkpoint':'./MODELS/old/n4_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_5$D-NN','checkpoint':'./MODELS/old/n5_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}
        ]
if wfs.routine == 'n4old_all_nZM24':# 2
    routine = [ 
        {'tag':'D-MVM','checkpoint':'./MODELS/old/n0_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_4$D-MVM','checkpoint':'./MODELS/old/n4_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'$\Lambda_4$-NN','checkpoint':'./MODELS/old/n4_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'D-NN','checkpoint':'./MODELS/old/n0_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_4$D-NN','checkpoint':'./MODELS/old/n4_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        ] 
    color_pwfs = ['black']
if wfs.routine == 'n5old_all_nZM24':# 2
    routine = [ 
        {'tag':'D-MVM','checkpoint':'./MODELS/old/n0_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_5$D-MVM','checkpoint':'./MODELS/old/n5_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'$\Lambda_5$-NN','checkpoint':'./MODELS/old/n5_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'D-NN','checkpoint':'./MODELS/old/n0_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_5$D-NN','checkpoint':'./MODELS/old/n5_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        ]
    color_pwfs = ['black']
if wfs.routine == 'testold_nZM24_nD10k_b10':
    routine = [ 
        {'tag':'$\Lambda_2$D-NN','checkpoint':'./MODELS/old/n2_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},   
        {'tag':'$\Lambda_3$D-NN','checkpoint':'./MODELS/old/n3_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_4$D-NN','checkpoint':'./MODELS/old/n4_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_5$D-NN','checkpoint':'./MODELS/old/n5_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'D-NN','checkpoint':'./MODELS/old/n0_all_cte-nRr0-nPx2_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        ] s

# nD50k: THIS AND THE NORMAL ND50K GIVE THE SAME RESULTS
if wfs.routine == 'nsold_p_nZM24_nD50k_b10':# 2
    routine = [
        {'tag':'D-MVM','checkpoint':'./MODELS/old/n0_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        #{'tag':'$\Lambda_2$D-MVM','checkpoint':'./MODELS/old/n2_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},  
        #{'tag':'$\Lambda_3$D-MVM','checkpoint':'./MODELS/old/n3_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        #{'tag':'$\Lambda_4$D-MVM','checkpoint':'./MODELS/old/n4_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        #{'tag':'$\Lambda_5$D-MVM','checkpoint':'./MODELS/old/n5_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'}
        ] 
if wfs.routine == 'nsold_nn_nZM24_nD50k_b10':# 1
    routine = [
        {'tag':'$\Lambda_2$-NN','checkpoint':'./MODELS/old/n2_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'$\Lambda_3$-NN','checkpoint':'./MODELS/old/n3_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_4$-NN','checkpoint':'./MODELS/old/n4_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_5$-NN','checkpoint':'./MODELS/old/n5_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'}
        ] 
if wfs.routine == 'nsold_pnn_nZM24_nD50k_b10':# 2
    routine = [
        {'tag':'D-NN','checkpoint':'./MODELS/old/n0_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        #{'tag':'$\Lambda_2$D-NN','checkpoint':'./MODELS/old/n2_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},  
        #{'tag':'$\Lambda_3$D-NN','checkpoint':'./MODELS/old/n3_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        #{'tag':'$\Lambda_4$D-NN','checkpoint':'./MODELS/old/n4_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        #{'tag':'$\Lambda_5$D-NN','checkpoint':'./MODELS/old/n5_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}
        ] 
if wfs.routine == 'n4old_all_nZM24_nD50k_b10':# 2
    routine = [ 
        {'tag':'D-MVM','checkpoint':'./MODELS/old/n0_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_4$D-MVM','checkpoint':'./MODELS/old/n4_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'$\Lambda_4$-NN','checkpoint':'./MODELS/old/n4_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'D-NN','checkpoint':'./MODELS/old/n0_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_4$D-NN','checkpoint':'./MODELS/old/n4_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        ] 
    color_pwfs = ['black']
if wfs.routine == 'n5old_all_nZM24_nD50k_b10':# 2
    routine = [ 
        {'tag':'D-MVM','checkpoint':'./MODELS/old/n0_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_5$D-MVM','checkpoint':'./MODELS/old/n5_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'$\Lambda_5$-NN','checkpoint':'./MODELS/old/n5_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'D-NN','checkpoint':'./MODELS/old/n0_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_5$D-NN','checkpoint':'./MODELS/old/n5_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        ]
    color_pwfs = ['black'] 
if wfs.routine == 'testold_nZM24_nD50k_b10':
    routine = [ 
        {'tag':'$\Lambda_2$D-NN','checkpoint':'./MODELS/old/n2_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'$\Lambda_3$D-NN','checkpoint':'./MODELS/old/n3_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_4$D-NN','checkpoint':'./MODELS/old/n4_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_5$D-NN','checkpoint':'./MODELS/old/n5_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'D-NN','checkpoint':'./MODELS/old/n0_all_cte-nRr0-nPx2_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        ] 

# nZM24_nD10k_b10_cte_nRr1
if wfs.routine == 'ns_p_nZM24_nD10k_b10_cte_nRr1':# 2
    routine = [
        #{'tag':'D-MVM','checkpoint':'./MODELS/nD10k/b10/nZ24/n0_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_2$D-MVM','checkpoint':'./MODELS/nD10k/b10/nZ24/n2_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'$\Lambda_3$D-MVM','checkpoint':'./MODELS/nD10k/b10/nZ24/n3_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_4$D-MVM','checkpoint':'./MODELS/nD10k/b10/nZ24/n4_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_5$D-MVM','checkpoint':'./MODELS/nD10k/b10/nZ24/n5_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'}
        ] 
if wfs.routine == 'ns_nn_nZM24_nD10k_b10_cte_nRr1':# 2
    routine = [ 
        {'tag':'$\Lambda_2$-NN','checkpoint':'./MODELS/nD150k/b10/nZ24/n2_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'$\Lambda_3$-NN','checkpoint':'./MODELS/nD10k/b10/nZ24/n3_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_4$-NN','checkpoint':'./MODELS/nD10k/b10/nZ24/n4_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_5$-NN','checkpoint':'./MODELS/nD10k/b10/nZ24/n5_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'}
        ] 
if wfs.routine == 'ns_pnn_nZM24_nD10k_b10_cte_nRr1':# 2
    routine = [
        #{'tag':'D-NN','checkpoint':'./MODELS/nD10k/b10/nZ24/n0_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_2$D-NN','checkpoint':'./MODELS/nD10k/b10/nZ24/n2_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'$\Lambda_3$D-NN','checkpoint':'./MODELS/nD10k/b10/nZ24/n3_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_4$D-NN','checkpoint':'./MODELS/nD10k/b10/nZ24/n4_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_5$D-NN','checkpoint':'./MODELS/nD10k/b10/nZ24/n5_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}
        ] 
if wfs.routine == 'n4_all_nZM24_nD10k_b10_cte_nRr1':# 2
    routine = [ 
        {'tag':'D-MVM','checkpoint':'./MODELS/nD10k/b10/nZ24/n0_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_4$D-MVM','checkpoint':'./MODELS/nD10k/b10/nZ24/n4_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'$\Lambda_4$-NN','checkpoint':'./MODELS/nD10k/b10/nZ24/n4_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'D-NN','checkpoint':'./MODELS/nD10k/b10/nZ24/n0_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_4$D-NN','checkpoint':'./MODELS/nD10k/b10/nZ24/n4_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}
        ]     
if wfs.routine == 'n5_all_nZM24_nD10k_b10_cte_nRr1':
    routine = [ 
        {'tag':'D-MVM','checkpoint':'./MODELS/nD10k/b10/nZ24/n0_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_5$D-MVM','checkpoint':'./MODELS/nD10k/b10/nZ24/n5_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'$\Lambda_5$-NN','checkpoint':'./MODELS/nD10k/b10/nZ24/n5_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'D-NN','checkpoint':'./MODELS/nD10k/b10/nZ24/n0_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_5$D-NN','checkpoint':'./MODELS/nD10k/b10/nZ24/n5_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}
        ] 
if wfs.routine == 'test_nZM24_nD10k_b10_cte_nRr1':
    routine = [ 
        {'tag':'$\Lambda_2$D-NN','checkpoint':'./MODELS/nD10k/b10/nZ24/n2_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'$\Lambda_3$D-NN','checkpoint':'./MODELS/nD10k/b10/nZ24/n3_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_4$D-NN','checkpoint':'./MODELS/nD10k/b10/nZ24/n4_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_5$D-NN','checkpoint':'./MODELS/nD10k/b10/nZ24/n5_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'D-NN','checkpoint':'./MODELS/nD10k/b10/nZ24/n0_all_cte_nRr1_nD10k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        ] 


# nZM24_nD50k_b10_cte_nRr1
if wfs.routine == 'ns_p_nZM24_nD50k_b10_cte_nRr1':# 2
    routine = [
        #{'tag':'D-MVM','checkpoint':'./MODELS/nD50k/b10/nZ24/n0_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_2$D-MVM','checkpoint':'./MODELS/nD50k/b10/nZ24/n2_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'$\Lambda_3$D-MVM','checkpoint':'./MODELS/nD50k/b10/nZ24/n3_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_4$D-MVM','checkpoint':'./MODELS/nD50k/b10/nZ24/n4_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_5$D-MVM','checkpoint':'./MODELS/nD50k/b10/nZ24/n5_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'}
        ] 
if wfs.routine == 'ns_nn_nZM24_nD50k_b10_cte_nRr1':# 2
    routine = [ 
        {'tag':'$\Lambda_2$-NN','checkpoint':'./MODELS/nD50k/b10/nZ24/n2_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'$\Lambda_3$-NN','checkpoint':'./MODELS/nD50k/b10/nZ24/n3_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_4$-NN','checkpoint':'./MODELS/nD50k/b10/nZ24/n4_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_5$-NN','checkpoint':'./MODELS/nD50k/b10/nZ24/n5_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'}
        ] 
if wfs.routine == 'ns_pnn_nZM24_nD50k_b10_cte_nRr1':# 2
    routine = [
        #{'tag':'D-NN','checkpoint':'./MODELS/nD50k/b10/nZ24/n0_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_2$D-NN','checkpoint':'./MODELS/nD50k/b10/nZ24/n2_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'$\Lambda_3$D-NN','checkpoint':'./MODELS/nD50k/b10/nZ24/n3_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_4$D-NN','checkpoint':'./MODELS/nD50k/b10/nZ24/n4_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_5$D-NN','checkpoint':'./MODELS/nD50k/b10/nZ24/n5_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}
        ] 
if wfs.routine == 'n4_all_nZM24_nD50k_b10_cte_nRr1':# 2
    routine = [ 
        {'tag':'D-MVM','checkpoint':'./MODELS/nD50k/b10/nZ24/n0_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_4$D-MVM','checkpoint':'./MODELS/nD50k/b10/nZ24/n4_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'$\Lambda_4$-NN','checkpoint':'./MODELS/nD50k/b10/nZ24/n4_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'D-NN','checkpoint':'./MODELS/nD50k/b10/nZ24/n0_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_4$D-NN','checkpoint':'./MODELS/nD50k/b10/nZ24/n4_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}
        ]     
if wfs.routine == 'n5_all_nZM24_nD50k_b10_cte_nRr1':
    routine = [ 
        {'tag':'D-MVM','checkpoint':'./MODELS/nD50k/b10/nZ24/n0_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_5$D-MVM','checkpoint':'./MODELS/nD50k/b10/nZ24/n5_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'$\Lambda_5$-NN','checkpoint':'./MODELS/nD50k/b10/nZ24/n5_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'D-NN','checkpoint':'./MODELS/nD50k/b10/nZ24/n0_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_5$D-NN','checkpoint':'./MODELS/nD50k/b10/nZ24/n5_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}
        ] 
if wfs.routine == 'test_nZM24_nD50k_b10_cte_nRr1':
    routine = [ 
        {'tag':'$\Lambda_2$D-NN','checkpoint':'./MODELS/nD50k/b10/nZ24/n2_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'$\Lambda_3$D-NN','checkpoint':'./MODELS/nD50k/b10/nZ24/n3_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_4$D-NN','checkpoint':'./MODELS/nD50k/b10/nZ24/n4_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_5$D-NN','checkpoint':'./MODELS/nD50k/b10/nZ24/n5_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'D-NN','checkpoint':'./MODELS/nD50k/b10/nZ24/n0_all_cte_nRr1_nD50k_b10_nZ24-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        ]   
    



def forward_model(model,phi,vNoise):
    if (phi.is_complex() or (len(phi.shape)!=4)): raise ValueError('Input cannot be complex')
    de = model.DE if hasattr(model,'DE') else model.DE_dummy
    fourier_mask = UNZ(UNZ(torch.exp(1j*de),0),0) * model.pwfs.fourier_mask
    I = Propagation(model.pwfs, phi, fourier_mask=fourier_mask)# T[b,1,N,M]
    I = addNoise(I,vNoise)# Addnoise
    I_deg = I.view(I.shape[0],1,-1)# T[b,1,NM]
    if hasattr(model,'NN'):
        I = model.norm_I(I,model.norm_nn)# T[b,1,N,M]
        Zest = model.NN(I)# T[b,z]
    else:
        I0_v,PIMat = model.pwfs.Calibration(fourier_mask=fourier_mask, vNoise=[0.,0.,0.,1.], mInorm=1)
        mI_v = model.pwfs.mI(I[:,0,model.pwfs.idx],I0_v, dim=1, mInorm=1)
        Zest = ( PIMat @ mI_v.t() ).t()# T[z,b]-> T[b,z]      
    return Zest,I_deg


'''fig,axs = plt.subplots(1,3, figsize=(9,3))
# phi
norm = Normalize(vmin=torch.min(PHI[t:t+1,0:1,:,:]), vmax=torch.max(PHI[t:t+1,0:1,:,:]))
axs[0].imshow((PHI[t:t+1,0:1,:,:]).squeeze(), norm=norm)
axs[0].set_title('$\phi_{tur}$')
axs[0].axis('off')
axs[1].imshow((phi_corr).squeeze(), norm=norm)
axs[1].set_title('$\phi_{corr}$')
axs[1].axis('off')
axs[2].imshow((phi_res).squeeze(), norm=norm)
axs[2].set_title('$\phi_{res}$') 
axs[2].axis('off')
fig.suptitle(f'PWFS cl: D/r0={wfs.Dr0v[id_r]} | mo={mo["tag"]} | it={t}')
plt.savefig( verbose_path + f'/dpwfs_Dro{int(wfs.Dr0v[id_r])}_mo{id_mo}_t{t}.png', dpi=100, bbox_inches='tight')
plt.close(fig)'''