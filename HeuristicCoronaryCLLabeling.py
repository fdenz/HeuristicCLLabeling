import numpy as np
import os
from scipy.stats import mode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# NOT FOR COMMERCIAL USE
# Implementation of an heuristic coronary centerline labeling approach by Felix Denzinger
# Context described in https://openreview.net/forum?id=vVPMifME8b

def get_second_longest_segment(seg_list, longest_segment):
    overlap = []
    seglens = []
    for seg in seg_list:
        seg_len = len(seg)
        seglens.append(seg_len)
        buffer = np.zeros_like(longest_segment)
        buffer[:len(seg)] = seg
        overlapping_part = np.argwhere(np.linalg.norm(buffer - longest_segment, axis=1) > 0.01)[0][0]
        overlap.append(seg_len - overlapping_part)
    return np.argmax(np.array(overlap)), np.array(seglens)[np.argmax(np.array(overlap))] - np.max(np.array(overlap))


def label_CL(CL,mat,pat,transform = True,visualize = True):

    #calculate centerline spacing
    spacing = mode(np.linalg.norm(CL[1:]-CL[:-1],axis=1))[0][0]

    if transform:
        CL = np.hstack([CL, np.ones((CL.shape[0],1))])
        CL = np.dot(np.linalg.inv(mat), CL.T)[:3].T

    Aorta = CL[0] # The centerlines always start in the center of the aorta in our format
    cuts = np.concatenate([np.linalg.norm(CL[1:]-CL[:-1],axis=1),np.zeros((1))],0) # Within a centerline the spacing is roughly equal in our format. Larger jumps indicate the start of a new centerline

    branches = []
    branch = []
    idxs = []
    id = 0
    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    for i,p in enumerate(CL):
        if cuts[i] <1:
            idxs.append(id)
            branch.append(p)
        else:
            idxs.append(id)
            branch.append(p)
            #don't add branches outside of the range
            if np.linalg.norm(np.array(branch)[0] - Aorta) < 4/spacing:
                branches.append(np.array(branch))
                if visualize:
                    x = np.array(branch)[:,0]
                    y = np.array(branch)[:,1]
                    z = np.array(branch)[:,2]
                    ax.scatter(x, y, z, color='red', s=0.5)
                    ax.plot(x, y, z, color='r')
            branch = []
            id+=1


    branches.sort(key = lambda x: len(x))
    branches = branches[::-1]

    # optional pruning step
    '''
    pruned_branches = []
    for brindex in range(len(branches)):
        add = True
        for br2index in range(len(branches)-1-brindex):
            buffer_branch = np.zeros_like(branches[brindex])
            buffer_branch[:len(branches[br2index])] = branches[br2index]
            diff_branch = np.linalg.norm(branches[brindex]-buffer_branch,axis=1)
            unique = len(diff_branch[diff_branch>0.5])
            if unique < 10:
                add = False
        if add:
            pruned_branches.append(branches[br2index])
    branches = pruned_branches
    '''

    labeled_branches = {}
    subseg_length = int(32/spacing)

    # Get the directions of the centerlines starting from the aortic center.
    directions = []
    distance = int(2/spacing)
    for b in branches:
        direction = b[distance,1]-b[0,1]
        directions.append(direction)

    unique_dirs = np.unique(np.array(directions))
    RAMUS = None
    LAD = None
    CX = None
    LAD_d1 = None
    CX_d1 = None
    branches = np.array(branches,dtype=object)
    if len(unique_dirs) >1:
        left_branches = branches[np.array(directions)==unique_dirs[-1]]
        right_branches = branches[np.array(directions)==unique_dirs[0]]
    else:
        # Catch cases where the left or right coronary tree was not extracted
        if unique_dirs[0]<0:
            right_branches = branches[np.array(directions)==unique_dirs[0]]
            left_branches = []
        else:
            left_branches = branches[np.array(directions) == unique_dirs[0]]
            right_branches = []

    if len(right_branches)>0:
        #right side
        RCA_lengths = [len(b) for b in right_branches]
        RCA = right_branches[np.argmax(np.array(RCA_lengths))]
        RCA_not_longus = right_branches[np.argmax(np.array(RCA_lengths)) != np.arange(len(right_branches))]

        try:
            RCA_least_overlap, cut_off = get_second_longest_segment(RCA_not_longus, RCA)
            # in theory one could also extract d1 of the RCA
            RCA_d1 = RCA_not_longus[RCA_least_overlap][cut_off:]
        except ValueError:
            RCA_d1 = None

        gradient = (RCA[:-1] - RCA[1:]).astype(float)
        gradient_gradient = np.linalg.norm(gradient[:-1] - gradient[1:], axis=1)
        ostium = np.argwhere(np.abs(gradient_gradient) > 0.01)[0][0]
        RCA = RCA[ostium:]
        for subbranch in ['RCA_PROX', 'RCA_MID', 'RCA_DIST']:
            dist = min(subseg_length, RCA.shape[0])
            labeled_branches[subbranch] = RCA[:dist]
            RCA = RCA[dist:]


    if len(left_branches)>0:
        #left side
        gradient = ((left_branches[0][:-1] - left_branches[0][1:]).astype(float))
        gradient_gradient = np.linalg.norm(gradient[:-1] - gradient[1:], axis=1)
        left_ostium = np.argwhere(np.abs(gradient_gradient) > 0.01)[0][0]
        left_branches = np.array([b[left_ostium:] for b in left_branches],dtype=object)

        left_branches = list(left_branches)
        left_branches.sort(key = lambda x: len(x))
        left_branches = left_branches[::-1]

        bifur = []
        for enu, b_to_prune in enumerate(left_branches):
            for b_to_compare in left_branches[enu + 1:]:
                buffer = np.zeros_like(b_to_prune)
                buffer[:b_to_compare.shape[0]] = b_to_compare
                difference = np.linalg.norm(b_to_prune - buffer, axis=1)
                mask = difference < 0.1
                if mask.sum() > 0:
                    point_after_cutoff = np.argwhere(mask == True)[-1] + 1
                    bifur.append(point_after_cutoff[0])
        cofs,counts = np.unique(np.array(bifur),return_counts=True)
        if len(counts)<1:
            return
        print(cofs,counts)
        bifur = cofs[np.argmax(counts)]
        if bifur > 24:
            labeled_branches['LM'] = left_branches[0][:bifur]
        distance = int(5/spacing)
        distance_far = int(12 / spacing)
        left_branches = [b for b in left_branches if len(b)>distance_far+bifur]
        left_branches = np.array(left_branches,dtype=object)


        directions = []
        for b in left_branches:
            from_lm = b[bifur:]
            direction = from_lm[distance]-from_lm[0]
            directions.append(direction)


        directions_far = []
        for b in left_branches:
            from_lm = b[bifur:]
            dist = min(distance_far,b.shape[0])
            direction_far = from_lm[dist] - from_lm[0]
            directions_far.append(direction_far)

        x_dirs = [dire[1] for dire in directions]
        y_dirs = [dire[0] for dire in directions_far]

        unique_dirs = np.unique(np.array(x_dirs)[np.array(y_dirs)>0])
        left_branches = np.array(left_branches,dtype=object)

        if len(unique_dirs) == 1:
            unique_dirs = np.unique(np.array(x_dirs)[np.array(y_dirs) > 0])

            LAD = left_branches[x_dirs==unique_dirs[0]]
            LAD_lengths = [len(b) for b in LAD]
            LAD_2 = LAD[np.argmax(np.array(LAD_lengths))!=np.arange(len(LAD))]
            LAD = LAD[np.argmax(np.array(LAD_lengths))]
            try:
                LAD_least_overlap,cut_off = get_second_longest_segment(LAD_2,LAD)
                LAD_d1 = LAD_2[LAD_least_overlap][cut_off:]
                try:
                    LAD_3 = LAD_2[LAD_least_overlap!=np.arange(len(LAD_2))]
                    LAD_least_overlap,cut_off = get_second_longest_segment(LAD_3,LAD)
                    LAD_d2 = LAD_3[LAD_least_overlap][cut_off:]
                except ValueError:
                    LAD_d2 = None
            except ValueError:
                LAD_d1 = None
            CX_d1 = None
            CX = None
            if unique_dirs[0]<0:
                CX = LAD
                LAD = None
                CX_d1 = LAD_d1
                LAD_d1 = None
            RAMUS = None


        if len(unique_dirs) == 2:
            LAD = left_branches[x_dirs==unique_dirs[0]]
            LAD_lengths = [len(b) for b in LAD]
            LAD_2 = LAD[np.argmax(np.array(LAD_lengths))!=np.arange(len(LAD))]
            LAD = LAD[np.argmax(np.array(LAD_lengths))]

            try:
                LAD_least_overlap,cut_off = get_second_longest_segment(LAD_2,LAD)
                LAD_d1 = LAD_2[LAD_least_overlap][cut_off:]
                try:
                    LAD_3 = LAD_2[LAD_least_overlap!=np.arange(len(LAD_2))]
                    LAD_least_overlap,cut_off = get_second_longest_segment(LAD_3,LAD)
                    LAD_d2 = LAD_3[LAD_least_overlap][cut_off:]
                except ValueError:
                    LAD_d2 = None
            except ValueError:
                LAD_d1 = None

            CX = left_branches[x_dirs==unique_dirs[1]]
            CX_lengths = [len(b) for b in CX]
            CX_2 = CX[np.argmax(np.array(CX_lengths))!=np.arange(len(CX))]
            CX = CX[np.argmax(np.array(CX_lengths))]
            try:
                CX_least_overlap,cut_off = get_second_longest_segment(CX_2,CX)
                CX_d1 = CX_2[CX_least_overlap][cut_off:]
            except ValueError:
                CX_d1 = None
            RAMUS = None


        elif len(unique_dirs) > 2:
            LAD = left_branches[x_dirs==unique_dirs[0]]
            LAD_lengths = [len(b) for b in LAD]
            LAD_2 = LAD[np.argmax(np.array(LAD_lengths))!=np.arange(len(LAD))]
            LAD = LAD[np.argmax(np.array(LAD_lengths))]
            try:
                LAD_least_overlap, cut_off = get_second_longest_segment(LAD_2, LAD)
                LAD_d1 = LAD_2[LAD_least_overlap][cut_off:]
                try:
                    LAD_3 = LAD_2[LAD_least_overlap!=np.arange(len(LAD_2))]
                    LAD_least_overlap,cut_off = get_second_longest_segment(LAD_3,LAD)
                    LAD_d2 = LAD_3[LAD_least_overlap][cut_off:]
                except ValueError:
                    LAD_d2 = None
            except ValueError:
                LAD_d1 = None

            RAMUS = left_branches[x_dirs==unique_dirs[len(unique_dirs)//2]]
            RAMUS_lengths = [len(b) for b in RAMUS]
            RAMUS = RAMUS[np.argmax(np.array(RAMUS_lengths))]


            CX = left_branches[x_dirs==unique_dirs[-1]]
            CX_lengths = [len(b) for b in CX]
            CX_2 = CX[np.argmax(np.array(CX_lengths))!=np.arange(len(CX))]
            CX = CX[np.argmax(np.array(CX_lengths))]
            try:
                CX_least_overlap, cut_off = get_second_longest_segment(CX_2, CX)
                CX_d1 = CX_2[CX_least_overlap][cut_off:]
            except ValueError:
                CX_d1 = None


        if RAMUS is not None:
            RAMUS = RAMUS[bifur:]
            dist = min(subseg_length, RAMUS.shape[0])
            # only include relevant rami intermedii
            if RAMUS.shape[0] >128:
                labeled_branches['RAMUS'] = RAMUS[:dist]
        if LAD is not None:
            LAD = LAD[bifur:]
            for subbranch in ['LAD_PROX', 'LAD_MID', 'LAD_DIST']:
                dist = min(subseg_length, LAD.shape[0])
                labeled_branches[subbranch] = LAD[:dist]
                LAD = LAD[dist:]
        if CX is not None:
            CX = CX[bifur:]
            for subbranch in ['CX_PROX', 'CX_DIST', 'CX_OM2']:
                dist = min(subseg_length, CX.shape[0])
                labeled_branches[subbranch] = CX[:dist]
                CX = CX[dist:]

        if LAD_d1 is not None:
            dist = min(subseg_length, LAD_d1.shape[0])
            labeled_branches['LAD_D1'] = LAD_d1[:dist]

        LAD_d2 = None
        if LAD_d2 is not None:
            dist = min(subseg_length, LAD_d2.shape[0])
            labeled_branches['LAD_D2'] = LAD_d2[:dist]

        if CX_d1 is not None:
            dist = min(subseg_length, CX_d1.shape[0])
            labeled_branches['CX_OM1'] = CX_d1[:dist]

    #if RCA_d1 is not None:
    #    dist = min(subseg_length, RCA_d1.shape[0])
    #    labeled_branches['RCA_d1'] = RCA_d1[:dist]

    branches_new = []
    branch_names = ['RCA_PROX', 'RCA_MID', 'RCA_DIST', 'CX_PROX', 'CX_DIST', 'CX_OM2', 'LAD_PROX', 'LAD_MID', 'LAD_DIST',
                'RAMUS', 'LM','LAD_D1','CX_OM1']#,'LAD_D2']#,'RCA_d1']



    id = -1
    idxs = []

    # if less than 8 subsegments are found usually the centerline extraction failed due to bad image quality
    if len(labeled_branches.keys())<8:
        print(labeled_branches.keys())
        return
    if not os.path.exists(pat):
        os.mkdir(pat)


    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for subbranch in labeled_branches.keys():
            branch_markers = labeled_branches[subbranch]
            x = branch_markers[:,0]
            y = branch_markers[:,1]
            z = branch_markers[:,2]
            ax.plot(x, y, z)
        plt.show()

    for subbranch in branch_names:
        id += 1
        try:
            branches_new.append(labeled_branches[subbranch])
            tags = np.zeros((labeled_branches[subbranch].shape[0])).astype(str)

            sub_b_transformed = np.hstack([labeled_branches[subbranch], np.ones((labeled_branches[subbranch].shape[0], 1))])
            sub_b_transformed = np.dot(mat, sub_b_transformed.T)[:3].T

            np.save(os.path.join(pat,branch_names[id]+'.npy'),sub_b_transformed)

            tags[:] = branch_names[id]
            idxs.append(tags)
        except KeyError:
            continue


fn = 'centerlines.npy'
CL = np.load(fn)
fn = 'world_mat.npy'
mat = np.load(fn).reshape(4,4)
label_CL(CL,mat,'P34',True)