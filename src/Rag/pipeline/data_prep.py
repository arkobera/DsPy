import pandas as pd

DB = 'src/Rag/archive/WinnersInterviewBlogPosts.csv'

def prepare_data():
    df = pd.read_csv(DB)
    blogs = df['content'].tolist()
    with open ('src/Rag/archive/blogs.txt','w') as f:
        for blog in blogs:
            f.write(blog)

if __name__ == "__main__":
    prepare_data()
    print("Blogs Extracted and Saved")