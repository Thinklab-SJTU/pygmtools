#include <functional>
#include <queue>

struct TreeNode
{
    std::pair<std::vector<long>, std::vector<long> > x_indices;
    double gplsh;
    long idx;
    TreeNode();
    TreeNode(const int &);
    TreeNode(const std::pair<std::vector<long>, std::vector<long> > &, const double &, const long &);
    bool operator>(const TreeNode &) const;
};

TreeNode::TreeNode()
{
    this->x_indices = std::pair<std::vector<long>, std::vector<long> >();
    this->gplsh = 0;
    this->idx = 0;
}

TreeNode::TreeNode(const int & len)
{
    this->x_indices = std::pair<std::vector<long>, std::vector<long> >(std::vector<long>(len), std::vector<long>(len));
    this->gplsh = 0;
    this->idx = 0;
}

TreeNode::TreeNode(const std::pair<std::vector<long>, std::vector<long> > &x_indices, const double &gplsh, const long &idx)
{
    this->x_indices = x_indices;
    this->gplsh = gplsh;
    this->idx = idx;
}

bool TreeNode::operator>(const TreeNode &c) const
{
    return this->gplsh > c.gplsh;
}

using tree_node_priority_queue = std::priority_queue<TreeNode, std::vector<TreeNode>, std::greater<TreeNode> >;
