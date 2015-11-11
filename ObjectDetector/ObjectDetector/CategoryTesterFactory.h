#include "CategoryTester.h"

namespace od
{
	class CategoryTesterFactory
	{
	public:

		enum TestMode{
			INTER_CATEGORY_TEST_MODE,
			INTRA_CATEGORY_TEST_MODE
		};

		static CategoryTester* getCategoryTester(TestMode test);
	};
}