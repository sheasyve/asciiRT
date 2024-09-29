# cuda_project
#### For now, a basic c++ implementation has been started in the cppapplication folder.
#### Installing this with docker should be somewhat simple for any operating system, so we dont need to use colab just yet.

## Project Best Practices (rough guideline)

## Branching
- **Main Branch (`main`)**:  
  For now the main branch contains starter code that does not work. Once we have a version 1.0,
  this branch should always work and contain code we know isn’t broken. Don’t commit directly to `main` to save your work unless it's a small, safe change.
  
- **Feature Branches**:  
  Create a new branch for any bigger changes or experiments. 
  - Branch naming can be simple, like `feature/your-name` or `fix/small-change`.

## Merging

- **Merging into `main`**:  
  Once it is up and running at least, only merge changes that are tested (at least make sure it runs). Make sure your changes won’t break anything important. If you're not 100% sure, ask one of the others to take a quick look.

## Communication
- **Avoid Overlap**:  
  Let everyone know what you’re working on to avoid two people doing the same thing. Just a quick message or note is enough.
  
- **Push Regularly**:  
  Push your changes frequently but keep things clean.

## Avoid Accidents
- **Backup**:  
  If you're making a big change, create a branch first – even if you think it's minor. This way, we can always go back if needed.
  
- **No Accidental Overwrites**:  
  Before pushing to `main`, make sure you're not overwriting anything critical. Check with others if you're unsure.

## Docker
- **Keep Docker in Mind**:  
  Make sure changes work with our Docker setup so things remain easy for everyone, regardless of the OS.
