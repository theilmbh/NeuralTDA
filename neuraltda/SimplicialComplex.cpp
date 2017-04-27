#include <list>
#include <set>
#include <vector>
#include <iostream>
#include <algorithm>

class Simplex {
public:
	std::vector<int> vertices;
	int dim;
	Simplex();
	Simplex(std::vector<int> verts);
	~Simplex();
	bool operator==(Simplex simplexB) const;
	bool operator!=(Simplex simplexB) const;
	bool operator<(Simplex simplexB) const;
	void print_vertices() const;
};

Simplex::Simplex()
{
	this->dim = -1;
}

Simplex::Simplex(std::vector<int> verts)
{	
	this->vertices.insert(this->vertices.begin(), verts.begin(), verts.end());
	this->dim = this->vertices.size()-1;
}

Simplex::~Simplex()
{

}

bool Simplex::operator==(Simplex simplexB) const
{
	return (this->vertices) == (simplexB.vertices);
}

bool Simplex::operator!=(Simplex simplexB) const
{
	return !(this->vertices == simplexB.vertices);
}

bool Simplex::operator<(Simplex simplexB) const
{
	if(this->dim != simplexB.dim)
	{
		return 0;
	}
	return this->vertices < simplexB.vertices;
}

void Simplex::print_vertices() const
{
	std::vector<int>::const_iterator it;
	it = this->vertices.begin();
	for(; it != this->vertices.end(); ++it)
	{
		std::cout << *it << ' ';
		
	}
	std::cout << '\n';
}

class SimplicialComplex
{
	typedef std::set<Simplex> chain_group;
	typedef std::vector<SimplicialComplex::chain_group> scg;

	SimplicialComplex::scg simplicial_chain_groups;
public:
	SimplicialComplex(std::vector<Simplex> K);
	~SimplicialComplex();
	void add_max_simplex(std::vector<Simplex> K);
	void print_scgs();
private:
	std::vector<Simplex> primary_faces(Simplex Q);
};

SimplicialComplex::SimplicialComplex(std::vector<Simplex> K)
{
	int max_dim = 0;
	for(int i=0; i<K.size(); i++)
	{
		max_dim = std::max(max_dim, K[i].dim);
	}

	for(int i=0; i<max_dim+1; i++)
	{
		this->simplicial_chain_groups.push_back(std::set<Simplex>());
	}
	add_max_simplex(K);
}	

SimplicialComplex::~SimplicialComplex()
{

}

void SimplicialComplex::add_max_simplex(std::vector<Simplex> K2)
{
	int k;
	Simplex Q;
	std::vector<Simplex> L;
	std::vector<Simplex>::iterator it;
	std::vector<Simplex> K(K2);
	SimplicialComplex::chain_group cg;
	while(!K.empty())
	{
		Q = K.back();
		K.pop_back();
		k = Q.dim;
		if(k <= 0) { continue; }
		L = primary_faces(Q);
		K.insert(K.begin(), L.begin(), L.end());
		std::sort(K.begin(), K.end());
		it = std::unique(K.begin(), K.end());
		K.resize(std::distance(K.begin(), it));
		this->simplicial_chain_groups[k-1].insert(L.begin(), L.end());
		this->simplicial_chain_groups[k].insert(Q);
	}
}

std::vector<Simplex> SimplicialComplex::primary_faces(Simplex Q)
{
	int dim = Q.dim;

	std::vector<int> new_verts;
	std::vector<int> pts = Q.vertices;
	std::vector<Simplex> L;
	pts.insert(pts.end(), pts.begin(), pts.end()-2);

	if(dim == 0)
	{
		return L;
	}

	for(int i=0; i <= dim; i++)
	{
		new_verts.insert(new_verts.begin(), pts.begin()+i, pts.begin()+(i+dim));
		std::sort(new_verts.begin(), new_verts.end());
		Simplex new_simplex = Simplex(new_verts);
		L.push_back(new_simplex);
		new_verts.clear();	
	}
	return L;
}

void SimplicialComplex::print_scgs()
{
	int dim = 0;
	SimplicialComplex::chain_group::iterator itbeg, itend;

	for(; dim < this->simplicial_chain_groups.size(); ++dim)
	{
		std::cout << "DIMENSION: " << dim << '\n';
		std::cout << "-----------\n";
		itbeg = this->simplicial_chain_groups[dim].begin();
		itend = this->simplicial_chain_groups[dim].end();
		for(; itbeg != itend; ++itbeg)
		{
			(*itbeg).print_vertices();
		}
		std::cout << '\n';
	}
}

int main()
{
	std::vector<int> mylist;
	std::vector<int> mylist2, mylist3;
  	std::vector<int>::iterator it;

  // set some initial values:
  	for (int i=1; i<=5; ++i) mylist.push_back(i); // 1 2 3 4 5
  	for (int i=1; i<=8; ++i) mylist2.push_back(i); // 1 2 3 4 5
  	for (int i=22; i<=40; ++i) mylist3.push_back(i); // 1 2 3 4 5
	Simplex simplex1 =  Simplex(mylist);
	Simplex simplex2 = Simplex(mylist2);
	Simplex simplex3 = Simplex(mylist3);
	if(simplex1 == simplex2){
		std::cout << "True\n";
	} else {
		std::cout << "False\n";
	}

	if(simplex1 == simplex3){
		std::cout << "True\n";
	} else {
		std::cout << "False\n";
	}
	if(mylist == mylist3){
		std::cout << "True\n";
	} else {
		std::cout << "False\n";
	}
	simplex1.print_vertices();
	simplex2.print_vertices();
	simplex3.print_vertices();

	std::vector<Simplex> mysc;
	mysc.push_back(simplex1);
	mysc.push_back(simplex2);
	//mysc.push_back(simplex3);
	SimplicialComplex mysc2 (mysc);
	mysc2.print_scgs();
	return 0;
}