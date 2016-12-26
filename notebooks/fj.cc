#include <iostream>
#include "fastjet/ClusterSequence.hh"

using namespace std;
using namespace fastjet;


void _traverse_rec(PseudoJet root, int parent_id, bool is_left,
                  vector<int>& tree, vector<double>& content){
    int id = tree.size() / 2;

    if (parent_id >= 0) {
        if (is_left) {
            tree[2 * parent_id] = id;
        } else {
            tree[2 * parent_id + 1] = id;
        }
    }

    tree.push_back(-1);
    tree.push_back(-1);
    content.push_back(root.px());
    content.push_back(root.py());
    content.push_back(root.pz());
    content.push_back(root.e());
    content.push_back(root.user_index());  // remove this for jet studies

    if (root.has_pieces()) {
        vector<PseudoJet> pieces = root.pieces();
        _traverse_rec(pieces[0], id, true, tree, content);
        _traverse_rec(pieces[1], id, false, tree, content);
    }
}

pair< vector<int>, vector<double> > _traverse(PseudoJet root){
    vector<int> tree;
    vector<double> content;
    _traverse_rec(root, -1, false, tree, content);
    return make_pair(tree, content);
}

static void fj(vector<double>& a,
               vector< vector<int> >& trees,
               vector< vector<double> >& contents,
               vector< double >& masses,
               vector< double >& pts,
               double R=1.0, int algorithm=0) {
    // Extract particles from array
    vector<fastjet::PseudoJet> particles;

    for (unsigned int i = 0; i < a.size(); i += 4) {
        fastjet::PseudoJet p = PseudoJet(a[i], a[i+1], a[i+2], a[i+3]);
        p.set_user_index((int) i / 4);
        particles.push_back(p);
    }

    // Cluster
    JetDefinition def(algorithm == 0 ? kt_algorithm : (algorithm == 1 ? antikt_algorithm : cambridge_algorithm), R);
    ClusterSequence seq(particles, def);
    vector<PseudoJet> jets = sorted_by_pt(seq.inclusive_jets());

    // Store results
    for (unsigned int j = 0; j < jets.size(); j++) {
        pair< vector<int>, vector<double> > p = _traverse(jets[j]);
        trees.push_back(p.first);
        contents.push_back(p.second);
        masses.push_back(jets[j].m());
        pts.push_back(jets[j].pt());
    }
}
