from mirror.celery import app
# from celery.decorators import shared_task

# logger=get_task_logger(__name__)

# This is the decorator which a celery worker uses
# @shared_task(name="test_task")
@app.task(bind=True)
def test_task(self, tenant, network_name):
    print("test_task executed", tenant, network_name)


# @app.task(bind=True)
# def test_task(self, job_name=None):

#     b = Tasks(task_id=self.request.id, job_name=job_name)
#     b.save()

#     self.update_state(state='Dispatching', meta={'progress': '33'})
#     sleep(random.randint(5, 10)) 

#     self.update_state(state='Running', meta={'progress': '66'})
#     sleep(random.randint(5, 10))  
#     self.update_state(state='Finishing', meta={'progress': '100'})
#     sleep(random.randint(5, 10)) 