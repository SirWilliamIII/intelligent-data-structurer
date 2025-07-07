#!/usr/bin/env python3
"""
Continue seeding the database with similar documents to trigger organic clustering
"""

import asyncio
import aiohttp

async def seed_for_organic_growth():
    """Upload many similar documents to trigger natural clustering."""
    
    print("🌱 Seeding Database for Organic Collection Growth")
    print("=" * 60)
    
    # More recipe documents (to cluster with existing ones)
    more_recipes = [
        {
            "filename": "pizza-dough.md",
            "content": """# Pizza Dough Recipe

## Ingredients
- 3 cups flour
- 1 cup warm water
- 2 tbsp olive oil
- 1 tsp salt
- 1 tsp yeast

## Instructions
1. Mix yeast with warm water
2. Add flour and salt
3. Knead for 10 minutes
4. Let rise 1 hour"""
        },
        {
            "filename": "pancakes.md",
            "content": """# Fluffy Pancakes

## Ingredients
- 2 cups flour
- 2 tbsp sugar
- 2 tsp baking powder
- 1 cup milk
- 2 eggs

## Instructions
1. Mix dry ingredients
2. Whisk wet ingredients
3. Combine and cook on griddle
4. Serve with syrup"""
        },
        {
            "filename": "meatloaf.md",
            "content": """# Classic Meatloaf

## Ingredients
- 2 lbs ground beef
- 1 cup breadcrumbs
- 2 eggs
- 1 onion diced
- 1/2 cup ketchup

## Instructions
1. Preheat oven to 375°F
2. Mix all ingredients
3. Shape into loaf
4. Bake 1 hour"""
        },
        {
            "filename": "caesar-salad.md",
            "content": """# Caesar Salad

## Ingredients
- 1 head romaine lettuce
- 1/2 cup parmesan cheese
- 1/4 cup caesar dressing
- 1 cup croutons
- Black pepper

## Instructions
1. Chop lettuce
2. Add dressing and toss
3. Top with cheese and croutons
4. Season with pepper"""
        },
        {
            "filename": "chocolate-cake.md",
            "content": """# Chocolate Cake

## Ingredients
- 2 cups flour
- 2 cups sugar
- 3/4 cup cocoa powder
- 2 eggs
- 1 cup milk

## Instructions
1. Preheat oven to 350°F
2. Mix dry ingredients
3. Add wet ingredients
4. Bake 30 minutes"""
        }
    ]
    
    # More technical tutorials (to cluster with existing tech docs)
    more_tech = [
        {
            "filename": "aws-cli.md",
            "content": """# AWS CLI Guide

## Installation
```bash
pip install awscli
aws configure
```

## Basic Commands
```bash
aws s3 ls
aws ec2 describe-instances
aws lambda list-functions
```

Essential cloud management commands."""
        },
        {
            "filename": "terraform-basics.md",
            "content": """# Terraform Tutorial

## Getting Started
```bash
terraform init
terraform plan
terraform apply
```

## Resource Definition
```hcl
resource "aws_instance" "web" {
  ami = "ami-12345"
  instance_type = "t2.micro"
}
```

Infrastructure as code fundamentals."""
        },
        {
            "filename": "nginx-config.md",
            "content": """# Nginx Configuration

## Basic Setup
```nginx
server {
    listen 80;
    server_name example.com;
    
    location / {
        proxy_pass http://localhost:3000;
    }
}
```

## SSL Configuration
```bash
sudo certbot --nginx
```

Web server configuration guide."""
        },
        {
            "filename": "mongodb-queries.md",
            "content": """# MongoDB Query Guide

## Basic Operations
```javascript
db.users.find({name: "John"})
db.users.insertOne({name: "Jane"})
db.users.updateOne({_id: id}, {$set: {age: 25}})
```

## Aggregation
```javascript
db.users.aggregate([
  {$match: {age: {$gte: 18}}},
  {$group: {_id: "$city", count: {$sum: 1}}}
])
```

Database query essentials."""
        },
        {
            "filename": "redis-commands.md",
            "content": """# Redis Cheat Sheet

## String Operations
```bash
SET key value
GET key
INCR counter
```

## List Operations
```bash
LPUSH mylist item
RPOP mylist
LRANGE mylist 0 -1
```

In-memory data structure commands."""
        }
    ]
    
    # Programming language guides (another potential cluster)
    programming_docs = [
        {
            "filename": "python-basics.md",
            "content": """# Python Fundamentals

## Variables and Types
```python
name = "John"
age = 30
is_student = True
```

## Control Flow
```python
if age >= 18:
    print("Adult")
else:
    print("Minor")
```

Essential Python programming concepts."""
        },
        {
            "filename": "javascript-es6.md",
            "content": """# JavaScript ES6 Features

## Arrow Functions
```javascript
const add = (a, b) => a + b;
const users = data.map(user => user.name);
```

## Destructuring
```javascript
const {name, age} = user;
const [first, second] = array;
```

Modern JavaScript syntax and features."""
        },
        {
            "filename": "go-tutorial.md",
            "content": """# Go Programming Guide

## Basic Syntax
```go
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
```

## Goroutines
```go
go func() {
    fmt.Println("Concurrent execution")
}()
```

Go language fundamentals and concurrency."""
        },
        {
            "filename": "rust-ownership.md",
            "content": """# Rust Ownership

## Ownership Rules
```rust
let s = String::from("hello");
let s2 = s; // s is no longer valid
```

## Borrowing
```rust
fn calculate_length(s: &String) -> usize {
    s.len()
}
```

Memory safety without garbage collection."""
        },
        {
            "filename": "sql-queries.md",
            "content": """# SQL Query Guide

## Basic Queries
```sql
SELECT * FROM users WHERE age > 18;
INSERT INTO users (name, email) VALUES ('John', 'john@email.com');
UPDATE users SET age = 25 WHERE id = 1;
```

## Joins
```sql
SELECT u.name, o.total 
FROM users u 
JOIN orders o ON u.id = o.user_id;
```

Database query fundamentals."""
        }
    ]
    
    # Sports/teams content (to cluster with existing football teams)
    sports_content = [
        {
            "filename": "basketball-teams.txt",
            "content": """NBA Teams

Los Angeles Lakers
Boston Celtics
Golden State Warriors
Chicago Bulls
Miami Heat
San Antonio Spurs
Brooklyn Nets"""
        },
        {
            "filename": "baseball-teams.txt",
            "content": """MLB Teams

New York Yankees
Los Angeles Dodgers
Boston Red Sox
Chicago Cubs
St. Louis Cardinals
Atlanta Braves
Houston Astros"""
        },
        {
            "filename": "soccer-teams.txt",
            "content": """Premier League Teams

Manchester United
Liverpool
Chelsea
Arsenal
Manchester City
Tottenham
Newcastle United"""
        },
        {
            "filename": "hockey-teams.txt",
            "content": """NHL Teams

Boston Bruins
Toronto Maple Leafs
Montreal Canadiens
New York Rangers
Chicago Blackhawks
Detroit Red Wings
Pittsburgh Penguins"""
        },
        {
            "filename": "tennis-players.txt",
            "content": """Tennis Champions

Roger Federer
Rafael Nadal
Novak Djokovic
Serena Williams
Venus Williams
Maria Sharapova
Andy Murray"""
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        
        all_groups = [
            ("🍳 More Recipes", more_recipes),
            ("💻 More Technical Docs", more_tech), 
            ("🔤 Programming Languages", programming_docs),
            ("⚽ Sports Content", sports_content)
        ]
        
        for group_name, docs in all_groups:
            print(f"\n{group_name} ({len(docs)} documents)")
            
            for i, doc in enumerate(docs):
                print(f"   📄 Uploading {doc['filename']} ({i+1}/{len(docs)})")
                
                data = aiohttp.FormData()
                data.add_field('file', 
                              doc['content'].encode('utf-8'),
                              filename=doc['filename'],
                              content_type='text/markdown')
                
                try:
                    async with session.post('http://localhost:8001/process', data=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            staging = result['ecosystem_status']['documents_in_staging']
                            seeds = result['ecosystem_status']['collection_seeds']
                            collections = result['ecosystem_status']['mature_collections']
                            
                            print(f"      ✅ Staged: {staging} | Seeds: {seeds} | Collections: {collections}")
                            
                            if seeds > 0:
                                print(f"      🌱 SEED DETECTED!")
                            if collections > 0:
                                print(f"      🎉 COLLECTION BORN!")
                                
                        else:
                            print(f"      ❌ Error: {response.status}")
                            
                except Exception as e:
                    print(f"      ❌ Error: {e}")
                
                # Small delay to let system process
                await asyncio.sleep(0.5)
            
            # Trigger evolution after each group
            print(f"   🧬 Triggering evolution...")
            try:
                async with session.post('http://localhost:8001/evolve') as response:
                    if response.status == 200:
                        result = await response.json()
                        ecosystem = result['ecosystem_after']
                        staging = ecosystem['ecosystem_status']['documents_in_staging']
                        seeds = ecosystem['ecosystem_status']['collection_seeds']
                        collections = ecosystem['ecosystem_status']['mature_collections']
                        
                        print(f"      📊 Post-evolution: Staging: {staging} | Seeds: {seeds} | Collections: {collections}")
                        
                        if ecosystem['collection_sizes']:
                            print(f"      🎉 COLLECTIONS FORMED:")
                            for name, size in ecosystem['collection_sizes'].items():
                                print(f"         • {name}: {size} documents")
                        
                        if ecosystem['suggestions']:
                            print(f"      💡 Suggestions: {len(ecosystem['suggestions'])}")
                            
            except Exception as e:
                print(f"   ❌ Evolution error: {e}")
        
        # Final comprehensive status
        print(f"\n📊 Final Ecosystem Status")
        print("=" * 40)
        
        try:
            async with session.get('http://localhost:8001/collections') as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"Total Collections: {len(result['collections'])}")
                    print(f"Total Documents: {result['total_documents']}")
                    print(f"Ecosystem Health: {result['ecosystem_health']:.1f}%")
                    
                    print(f"\nCollection Breakdown:")
                    for collection in result['collections']:
                        print(f"   📚 {collection['name']}: {collection['document_count']} docs ({collection['type']})")
                        if 'themes' in collection:
                            themes = ', '.join(collection['themes'][:3])
                            print(f"      🏷️  Themes: {themes}")
                        if 'health' in collection:
                            health = collection['health']
                            print(f"      🏥 Health: {health['coherence_score']:.1%} coherence")
                            
        except Exception as e:
            print(f"❌ Status error: {e}")
    
    print(f"\n✅ Database seeding complete!")
    print(f"🌐 View ecosystem: http://localhost:8001/collections")

if __name__ == "__main__":
    asyncio.run(seed_for_organic_growth())